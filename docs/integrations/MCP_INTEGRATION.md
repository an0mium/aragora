# MCP Integration Guide

Aragora provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that allows Claude Desktop and other MCP-compatible clients to run decision stress-tests and gauntlet red-team runs directly from conversations.

## Overview

The MCP server exposes Aragora's core capabilities as tools and resources:

### Tools

The MCP server exposes a growing set of tools. Current categories include:

- Core: `run_debate`, `run_gauntlet`, `list_agents`, `get_debate`, `search_debates`
- Agent stats: `get_agent_history`, `get_consensus_proofs`, `list_trending_topics`
- Memory: `query_memory`, `store_memory`, `get_memory_pressure`
- Forks: `fork_debate`, `get_forks`
- Genesis: `get_agent_lineage`, `breed_agents`
- Checkpoints: `create_checkpoint`, `list_checkpoints`, `resume_checkpoint`, `delete_checkpoint`
- Verification: `verify_consensus`, `generate_proof`
- Evidence: `search_evidence`, `cite_evidence`, `verify_citation`

See `aragora/mcp/tools.py` for the authoritative list and parameter schemas.

### Resources

| URI Template | Description |
|--------------|-------------|
| `debate://{debate_id}` | Access debate results by ID |
| `agent://{agent_name}/stats` | Access agent statistics and ELO rating |
| `consensus://{debate_id}` | Access formal verification proofs for a debate |
| `trending://topics` | Access current trending topics |

## Quick Start

### 1. Install Aragora

```bash
pip install aragora
```

### 2. Configure Claude Desktop

Add Aragora to your Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "aragora": {
      "command": "aragora",
      "args": ["mcp-server"],
      "env": {
        "ANTHROPIC_API_KEY": "your_key_here",
        "OPENAI_API_KEY": "your_key_here"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

Restart Claude Desktop to load the MCP server. You should see "aragora" in the available tools.

## Tools Reference

### run_debate

Run a decision stress-test (debate engine) on any topic.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | (required) | The question or topic to stress-test |
| `agents` | string | `"anthropic-api,openai-api"` | Comma-separated agent IDs |
| `rounds` | integer | `3` | Number of debate rounds (1-10) |
| `consensus` | string | `"majority"` | Consensus mechanism |

**Consensus mechanisms:**
- `majority` - Consensus when >50% of agents agree
- `unanimous` - Consensus requires all agents to agree
- `none` - No consensus detection, run all rounds

**Example usage in Claude:**
```
Use the run_debate tool to stress-test "Should we use microservices or a monolith for a 5-person startup?"
with agents anthropic-api,openai-api,gemini for 3 rounds.
```

**Example response:**
```json
{
  "debate_id": "mcp_a1b2c3d4",
  "task": "Should we use microservices or a monolith?",
  "final_answer": "For a 5-person startup, start with a well-structured monolith...",
  "consensus_reached": true,
  "confidence": 0.85,
  "rounds_used": 2,
  "agents": ["anthropic-api_proposer", "openai-api_critic", "gemini_synthesizer"]
}
```

### run_gauntlet

Stress-test specifications, code, or policies through adversarial analysis.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | (required) | Content to stress-test |
| `content_type` | string | `"spec"` | Type: spec, code, policy, architecture |
| `profile` | string | `"quick"` | Test profile |

**Profiles:**
| Profile | Duration | Focus |
|---------|----------|-------|
| `quick` | ~2 min | High-severity issues |
| `thorough` | ~15 min | Comprehensive analysis |
| `code` | ~10 min | Security + code quality |
| `security` | ~10 min | Security vulnerabilities |
| `gdpr` | ~10 min | GDPR compliance |
| `hipaa` | ~10 min | HIPAA compliance |

**Example usage in Claude:**
```
Use run_gauntlet to stress-test this API spec:

## User API
- POST /users - Create user
- GET /users/:id - Get user by ID
- DELETE /users/:id - Delete user

Use the security profile.
```

**Example response:**
```json
{
  "verdict": "CONDITIONAL",
  "risk_score": 0.45,
  "vulnerabilities_count": 3,
  "vulnerabilities": [
    {
      "category": "authentication",
      "severity": "high",
      "description": "No authentication specified for DELETE endpoint"
    }
  ]
}
```

### list_agents

Get a list of available AI agents.

**Example response:**
```json
{
  "agents": [
    "anthropic-api",
    "openai-api",
    "gemini",
    "grok",
    "mistral-api",
    "deepseek",
    "qwen"
  ],
  "count": 7
}
```

### get_debate

Retrieve results of a previous debate by ID.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `debate_id` | string | The debate ID to retrieve |

**Example usage:**
```
Use get_debate to retrieve debate mcp_a1b2c3d4
```

### search_debates

Search debates by topic, date range, or participating agents.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | - | Search query for topic text |
| `agent` | string | - | Filter by agent name |
| `start_date` | string | - | Start date (YYYY-MM-DD) |
| `end_date` | string | - | End date (YYYY-MM-DD) |
| `consensus_only` | boolean | `false` | Only return debates that reached consensus |
| `limit` | integer | `20` | Max results (1-100) |

**Example usage:**
```
Use search_debates to find debates about "microservices" that reached consensus
```

**Example response:**
```json
{
  "debates": [
    {
      "debate_id": "mcp_a1b2c3d4",
      "task": "Should we use microservices or a monolith?",
      "consensus_reached": true,
      "confidence": 0.85,
      "timestamp": "2025-01-12 10:30:00"
    }
  ],
  "count": 1,
  "query": "microservices",
  "filters": {
    "agent": null,
    "consensus_only": true,
    "date_range": "* to *"
  }
}
```

### get_agent_history

Get an agent's debate history, ELO rating, and performance stats.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | string | (required) | The agent name (e.g., 'anthropic-api') |
| `include_debates` | boolean | `true` | Include recent debate summaries |
| `limit` | integer | `10` | Max debates to include |

**Example usage:**
```
Use get_agent_history to see anthropic-api's performance stats and recent debates
```

**Example response:**
```json
{
  "agent_name": "anthropic-api",
  "elo_rating": 1650,
  "elo_deviation": 45,
  "total_debates": 127,
  "consensus_rate": 0.78,
  "win_rate": 0.62,
  "avg_confidence": 0.81,
  "recent_debates": [
    {
      "debate_id": "mcp_a1b2c3d4",
      "task": "Should we use microservices or a monolith?",
      "consensus_reached": true,
      "timestamp": "2025-01-12 10:30:00"
    }
  ]
}
```

### get_consensus_proofs

Retrieve formal verification proofs from debates.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debate_id` | string | - | Specific debate ID (optional, searches all if omitted) |
| `proof_type` | string | `"all"` | Type: z3, lean, or all |
| `limit` | integer | `10` | Max proofs to return |

**Example usage:**
```
Use get_consensus_proofs to get Z3 proofs from debate mcp_a1b2c3d4
```

**Example response:**
```json
{
  "proofs": [
    {
      "debate_id": "mcp_a1b2c3d4",
      "type": "z3",
      "statement": "consensus_valid",
      "proof": "(declare-const agreement Int)...",
      "verified": true
    }
  ],
  "count": 1,
  "debate_id": "mcp_a1b2c3d4",
  "proof_type": "z3"
}
```

### list_trending_topics

Get trending topics from Pulse that could make good debates.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform` | string | `"all"` | Source: hackernews, reddit, arxiv, all |
| `category` | string | - | Topic category filter (e.g., 'tech', 'ai') |
| `min_score` | number | `0.5` | Minimum topic score (0-1) |
| `limit` | integer | `10` | Max topics to return |

**Example usage:**
```
Use list_trending_topics to find AI-related topics from Hacker News with high debate potential
```

**Example response:**
```json
{
  "topics": [
    {
      "topic": "The future of AI regulation in the EU",
      "platform": "hackernews",
      "category": "ai",
      "score": 0.85,
      "volume": 342,
      "debate_potential": "high"
    }
  ],
  "count": 1,
  "platform": "hackernews",
  "category": "ai",
  "min_score": 0.5
}
```

## Resources Reference

Resources allow direct access to Aragora data via URI patterns.

### debate://{debate_id}

Access debate results directly.

**Example:**
```
Read the resource debate://mcp_a1b2c3d4
```

### agent://{agent_name}/stats

Access agent statistics and ELO rating.

**Example:**
```
Read the resource agent://anthropic-api/stats
```

### consensus://{debate_id}

Access formal verification proofs for a specific debate.

**Example:**
```
Read the resource consensus://mcp_a1b2c3d4
```

### trending://topics

Access current trending topics from Pulse.

**Example:**
```
Read the resource trending://topics
```

## Running the Server Manually

For debugging or development:

```bash
# Start the MCP server directly
python -m aragora.mcp.server

# Or via CLI
aragora mcp-server

# With debug logging
LOG_LEVEL=DEBUG aragora mcp-server
```

## Troubleshooting

### "No valid agents could be created"

Check that you have at least one API key set:
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

### Server not appearing in Claude Desktop

1. Check the config file path is correct for your OS
2. Verify JSON syntax is valid
3. Restart Claude Desktop completely (quit and reopen)
4. Check Claude Desktop logs for errors

### "Content is required" error

The `question` parameter for debates and `content` parameter for gauntlet are required. Make sure to provide them in your prompt.

### Rate limiting

If you hit rate limits, consider:
- Using fewer agents
- Reducing the number of rounds
- Using the `quick` gauntlet profile instead of `thorough`

### MCP package not installed

If you see "MCP package not installed":
```bash
pip install mcp
```

## API Key Requirements

| Agent | Required Key |
|-------|--------------|
| `anthropic-api` | `ANTHROPIC_API_KEY` |
| `openai-api` | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `grok` | `XAI_API_KEY` |
| `mistral-api` | `MISTRAL_API_KEY` |
| `codestral` | `MISTRAL_API_KEY` |
| `deepseek` | `OPENROUTER_API_KEY` |
| `mistral` | `OPENROUTER_API_KEY` |
| `qwen` | `OPENROUTER_API_KEY` |
| `qwen-max` | `OPENROUTER_API_KEY` |
| `llama` | `OPENROUTER_API_KEY` |
| `yi` | `OPENROUTER_API_KEY` |

At least one key is required. For best results, set both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

## Example Workflows

### Validate an API Design

```
I'm designing a user authentication API. Please use run_gauntlet with the security profile
to stress-test this design:

[paste your API spec here]
```

### Get Multiple AI Perspectives

```
Use run_debate to get perspectives on "What's the best approach to error handling
in a REST API?" Use anthropic-api,openai-api,gemini for diverse viewpoints.
```

### Review a Decision

```
Run a 5-round debate on "Should we migrate our database from PostgreSQL to MongoDB
for our social media application?" with consensus type unanimous.
```

### Find Debate-worthy Topics

```
Use list_trending_topics to find high-scoring AI topics from Hacker News,
then run a debate on the most interesting one.
```

### Compare Agent Performance

```
Use get_agent_history for anthropic-api and openai-api to compare their
ELO ratings and consensus rates.
```

### Search Past Debates

```
Use search_debates to find all debates about "architecture" from the past month
that reached consensus.
```

## Related Documentation

- [MCP Advanced Usage](./MCP_ADVANCED.md) - Advanced patterns and customization
- [Gauntlet Mode](./GAUNTLET.md) - Full gauntlet documentation
- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Environment Variables](./ENVIRONMENT.md) - All configuration options
