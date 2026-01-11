# MCP Integration Guide

Aragora provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that allows Claude Desktop and other MCP-compatible clients to run debates and gauntlet stress-tests directly from conversations.

## Overview

The MCP server exposes Aragora's core capabilities as tools:

| Tool | Description |
|------|-------------|
| `run_debate` | Run a multi-agent AI debate on a topic |
| `run_gauntlet` | Stress-test content through adversarial analysis |
| `list_agents` | List available AI agents |
| `get_debate` | Retrieve results of a previous debate |

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
      "args": ["mcp-server"]
    }
  }
}
```

### 3. Set Environment Variables

Ensure at least one AI provider API key is set:

```bash
export ANTHROPIC_API_KEY=your_key_here
# and/or
export OPENAI_API_KEY=your_key_here
```

### 4. Restart Claude Desktop

Restart Claude Desktop to load the MCP server. You should see "aragora" in the available tools.

## Tools Reference

### run_debate

Run a multi-agent debate on any topic.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | (required) | The question or topic to debate |
| `agents` | string | `"anthropic-api,openai-api"` | Comma-separated agent IDs |
| `rounds` | integer | `3` | Number of debate rounds (1-10) |
| `consensus` | string | `"majority"` | Consensus mechanism |

**Consensus mechanisms:**
- `majority` - Consensus when >50% of agents agree
- `unanimous` - Consensus requires all agents to agree
- `none` - No consensus detection, run all rounds

**Example usage in Claude:**
```
Use the run_debate tool to debate "Should we use microservices or a monolith for a 5-person startup?"
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
    "deepseek",
    "mistral"
  ],
  "count": 6
}
```

### get_debate

Retrieve results of a previous debate by ID.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `debate_id` | string | The debate ID to retrieve |

## Running the Server Manually

For debugging or development:

```bash
# Start the MCP server directly
python -m aragora.mcp.server

# Or via CLI
aragora mcp-server
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

## API Key Requirements

| Agent | Required Key |
|-------|--------------|
| `anthropic-api` | `ANTHROPIC_API_KEY` |
| `openai-api` | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `grok` | `XAI_API_KEY` |
| `deepseek` | `OPENROUTER_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |

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

## Related Documentation

- [Gauntlet Mode](./GAUNTLET.md) - Full gauntlet documentation
- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Environment Variables](./ENVIRONMENT.md) - All configuration options
