# Aragora MCP Module

Model Context Protocol (MCP) integration for Aragora. This module exposes Aragora's debate engine, gauntlet stress-testing, knowledge management, and control plane capabilities as MCP tools for Claude and other MCP-compatible clients.

## Overview

The MCP module provides:

- **70+ Tools** across 15 categories for debate, knowledge, workflow, and system operations
- **MCP Server** with rate limiting, input validation, and resource caching
- **Resource Templates** for accessing debate results, agent stats, and trending topics
- **Multi-instance Support** via Redis-backed rate limiting

## Quick Start

### Starting the MCP Server

```bash
# Via CLI
aragora mcp-server

# Or directly
python -m aragora.mcp.server
```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

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

## Architecture

```
aragora/mcp/
├── __init__.py           # Package exports
├── server.py             # MCP server implementation (AragoraMCPServer)
├── tools.py              # Tool registry and metadata
└── tools_module/         # Tool implementations by category
    ├── debate.py         # Core debate operations
    ├── gauntlet.py       # Document stress-testing
    ├── agent.py          # Agent management and breeding
    ├── memory.py         # Continuum memory operations
    ├── checkpoint.py     # Debate checkpoint management
    ├── verification.py   # Consensus verification and proofs
    ├── evidence.py       # Evidence collection and citation
    ├── trending.py       # Trending topic analysis (Pulse)
    ├── audit.py          # Document audit operations
    ├── knowledge.py      # Knowledge Mound operations
    ├── workflow.py       # Workflow execution
    ├── integrations.py   # External webhook integrations
    ├── control_plane.py  # Agent registry and task scheduling
    ├── canvas.py         # Visual canvas collaboration
    ├── browser.py        # Browser automation (Playwright)
    ├── context_tools.py  # Rich context fetching for chat
    └── chat_actions.py   # Tool-first chat interactions
```

## Available Tools

### Debate Tools

| Tool | Description |
|------|-------------|
| `run_debate` | Run a multi-agent AI debate on a topic |
| `get_debate` | Get results of a previous debate |
| `search_debates` | Search debates by topic, date, or agents |
| `fork_debate` | Fork a debate to explore counterfactual scenarios |
| `get_forks` | Get all forks of a debate |

**Example: Running a Debate**

```python
result = await run_debate_tool(
    question="Should we adopt microservices architecture?",
    agents="anthropic-api,openai-api,grok",
    rounds=3,
    consensus="majority"
)
# Returns: debate_id, final_answer, consensus_reached, confidence
```

### Gauntlet Tools

| Tool | Description |
|------|-------------|
| `run_gauntlet` | Stress-test content through adversarial analysis |

**Profiles:** `quick`, `thorough`, `code`, `security`, `gdpr`, `hipaa`

### Agent Tools

| Tool | Description |
|------|-------------|
| `list_agents` | List available AI agents |
| `get_agent_history` | Get agent debate history and performance stats |
| `get_agent_lineage` | Get the evolutionary lineage of an agent |
| `breed_agents` | Breed two agents to create a new offspring agent |

### Memory Tools

| Tool | Description |
|------|-------------|
| `query_memory` | Query memories from the continuum memory system |
| `store_memory` | Store a memory in the continuum memory system |
| `get_memory_pressure` | Get current memory pressure and utilization |

**Memory Tiers:** `fast` (1 min TTL), `medium` (1 hour), `slow` (1 day), `glacial` (1 week)

### Checkpoint Tools

| Tool | Description |
|------|-------------|
| `create_checkpoint` | Create a checkpoint for a debate to enable resume later |
| `list_checkpoints` | List checkpoints for a debate or all debates |
| `resume_checkpoint` | Resume a debate from a checkpoint |
| `delete_checkpoint` | Delete a checkpoint |

### Verification Tools

| Tool | Description |
|------|-------------|
| `get_consensus_proofs` | Retrieve formal verification proofs from debates |
| `verify_consensus` | Verify the consensus of a completed debate using formal methods |
| `generate_proof` | Generate a formal proof for a claim without verification |

**Backends:** `z3`, `lean4`

### Evidence Tools

| Tool | Description |
|------|-------------|
| `search_evidence` | Search for evidence across configured sources |
| `cite_evidence` | Add a citation to evidence in a debate message |
| `verify_citation` | Verify that a citation URL is valid and accessible |

### Trending Tools

| Tool | Description |
|------|-------------|
| `list_trending_topics` | Get trending topics from Pulse for debates |

**Platforms:** HackerNews, Reddit, Twitter (or `all`)

### Audit Tools

| Tool | Description |
|------|-------------|
| `list_audit_presets` | List available audit presets |
| `list_audit_types` | List registered audit types |
| `get_audit_preset` | Get details of a specific audit preset |
| `create_audit_session` | Create a new document audit session |
| `run_audit` | Start running an audit session |
| `get_audit_status` | Get status and progress of an audit session |
| `get_audit_findings` | Get findings from an audit session |
| `update_finding_status` | Update the workflow status of a finding |
| `run_quick_audit` | Run a quick audit using a preset |

**Presets:** Legal Due Diligence, Financial Audit, Code Security

**Finding Statuses:** `open`, `triaging`, `investigating`, `remediating`, `resolved`, `false_positive`, `accepted_risk`

### Knowledge Tools

| Tool | Description |
|------|-------------|
| `query_knowledge` | Query the Knowledge Mound for relevant information |
| `store_knowledge` | Store a new knowledge node in the Knowledge Mound |
| `get_knowledge_stats` | Get statistics about the Knowledge Mound |
| `get_decision_receipt` | Get a formal decision receipt for a completed debate |

**Node Types:** `fact`, `insight`, `claim`, `evidence`, `decision`, `opinion`

**Example: Querying Knowledge**

```python
result = await query_knowledge_tool(
    query="rate limiting strategies",
    node_types="fact,insight",
    min_confidence=0.7,
    limit=10,
    include_relationships=True
)
```

### Workflow Tools

| Tool | Description |
|------|-------------|
| `run_workflow` | Execute a workflow from a template |
| `get_workflow_status` | Get the status of a workflow execution |
| `list_workflow_templates` | List available workflow templates |
| `cancel_workflow` | Cancel a running workflow execution |

### Integration Tools

| Tool | Description |
|------|-------------|
| `trigger_external_webhook` | Trigger an external automation webhook (Zapier, Make, n8n) |
| `list_integrations` | List configured external integrations |
| `test_integration` | Test an integration connection |
| `get_integration_events` | Get available event types for an integration platform |

### Control Plane Tools

| Tool | Description |
|------|-------------|
| `register_agent` | Register an agent with the control plane |
| `unregister_agent` | Unregister an agent from the control plane |
| `list_registered_agents` | List all agents registered with the control plane |
| `get_agent_health` | Get detailed health status for a specific agent |
| `submit_task` | Submit a task to the control plane for execution |
| `get_task_status` | Get the status of a task in the control plane |
| `cancel_task` | Cancel a pending or running task |
| `list_pending_tasks` | List tasks in the pending queue |
| `get_control_plane_status` | Get overall control plane health and status |
| `trigger_health_check` | Trigger a health check for an agent or all agents |
| `get_resource_utilization` | Get resource utilization metrics |

**Task Priorities:** `low`, `normal`, `high`, `urgent`

### Canvas Tools

| Tool | Description |
|------|-------------|
| `canvas_create` | Create a new interactive canvas for visual collaboration |
| `canvas_get` | Get the state of a canvas including nodes and edges |
| `canvas_add_node` | Add a node to a canvas |
| `canvas_add_edge` | Add an edge between two nodes on a canvas |
| `canvas_execute_action` | Execute an action on a canvas |
| `canvas_list` | List available canvases |
| `canvas_delete_node` | Delete a node from a canvas |

**Node Types:** `text`, `agent`, `debate`, `knowledge`, `workflow`, `browser`, `input`, `output`

**Edge Types:** `default`, `data_flow`, `control_flow`, `reference`, `dependency`

**Actions:** `start_debate`, `run_workflow`, `query_knowledge`, `clear_canvas`

### Browser Automation Tools

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate the browser to a URL |
| `browser_click` | Click an element on the page |
| `browser_fill` | Fill a form field with a value |
| `browser_screenshot` | Capture a screenshot of the page or element |
| `browser_get_text` | Get the text content of an element |
| `browser_extract` | Extract data from multiple elements |
| `browser_execute_script` | Execute JavaScript in the browser context |
| `browser_wait_for` | Wait for an element to reach a specific state |
| `browser_get_html` | Get the HTML content of the page or element |
| `browser_close` | Close the browser session |
| `browser_get_cookies` | Get cookies from the browser session |
| `browser_clear_cookies` | Clear all cookies from the browser session |

**Requirements:** `pip install playwright && playwright install`

### Context Tools (Chat Integration)

| Tool | Description |
|------|-------------|
| `fetch_channel_context` | Fetch rich context from a chat channel |
| `fetch_debate_context` | Fetch context from an ongoing debate |
| `analyze_conversation` | Analyze a conversation for key themes |
| `get_thread_context` | Get context from a specific thread |
| `get_user_context` | Get context about a user's history |

### Chat Action Tools

| Tool | Description |
|------|-------------|
| `send_message` | Send a message to a chat channel |
| `create_poll` | Create a poll in a chat channel |
| `trigger_debate` | Trigger a debate from a chat message |
| `post_receipt` | Post a decision receipt to a channel |
| `update_message` | Update an existing message |
| `add_reaction` | Add a reaction to a message |
| `create_thread` | Create a new thread from a message |
| `stream_progress` | Stream progress updates to a channel |

## Creating Custom MCP Tools

### Tool Implementation Pattern

Create a new tool module in `tools_module/`:

```python
# aragora/mcp/tools_module/my_tools.py
"""
MCP Tools for my custom operations.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def my_custom_tool(
    required_param: str,
    optional_param: str = "default",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Do something useful.

    Args:
        required_param: Description of required parameter
        optional_param: Description of optional parameter
        limit: Maximum results to return

    Returns:
        Dict with operation results
    """
    if not required_param:
        return {"error": "required_param is required"}

    try:
        # Your implementation here
        result = await do_something(required_param, optional_param)

        return {
            "success": True,
            "data": result,
            "count": len(result),
        }
    except ImportError:
        return {"error": "Required module not available"}
    except Exception as e:
        logger.error(f"my_custom_tool failed: {e}")
        return {"error": f"Operation failed: {str(e)}"}


__all__ = ["my_custom_tool"]
```

### Registering Tools

1. **Export from `tools_module/__init__.py`:**

```python
from aragora.mcp.tools_module.my_tools import my_custom_tool

__all__ = [
    # ... existing exports
    "my_custom_tool",
]
```

2. **Add metadata to `tools.py`:**

```python
from aragora.mcp.tools_module import my_custom_tool

TOOLS_METADATA = [
    # ... existing tools
    {
        "name": "my_custom_tool",
        "description": "Do something useful",
        "function": my_custom_tool,
        "parameters": {
            "required_param": {"type": "string", "required": True},
            "optional_param": {"type": "string", "default": "default"},
            "limit": {"type": "integer", "default": 10},
        },
    },
]
```

### Parameter Schema

Supported parameter properties:

| Property | Description |
|----------|-------------|
| `type` | `string`, `integer`, `number`, `boolean` |
| `required` | `True` if the parameter is mandatory |
| `default` | Default value if not provided |
| `description` | Human-readable description |
| `enum` | List of allowed values |
| `minimum` | Minimum value for numbers |
| `maximum` | Maximum value for numbers |

## Configuration

### Environment Variables

**Rate Limiting:**

```bash
# Backend: "memory" (default) or "redis"
MCP_RATE_LIMIT_BACKEND=memory

# Redis URL (for redis backend)
MCP_REDIS_URL=redis://localhost:6379

# Per-tool rate limits (requests per minute)
MCP_RATE_LIMIT_RUN_DEBATE=10
MCP_RATE_LIMIT_RUN_GAUNTLET=20
MCP_RATE_LIMIT_LIST_AGENTS=60
MCP_RATE_LIMIT_GET_DEBATE=60
MCP_RATE_LIMIT_SEARCH_DEBATES=30
```

**Input Validation:**

```python
# Maximum input sizes (defined in server.py)
MAX_QUESTION_LENGTH = 10000
MAX_CONTENT_LENGTH = 100000
MAX_QUERY_LENGTH = 1000
```

### Rate Limiting

The server supports two rate limiting backends:

**In-Memory (default):**
- Suitable for single-instance deployments
- No external dependencies
- State lost on restart

**Redis:**
- Required for multi-instance deployments
- Shared state across instances
- Install: `pip install redis`

```python
from aragora.mcp.server import create_rate_limiter

# In-memory limiter
limiter = create_rate_limiter(backend="memory")

# Redis limiter
limiter = create_rate_limiter(
    backend="redis",
    redis_url="redis://localhost:6379",
    limits={"run_debate": 5}
)
```

## Resource Templates

The MCP server exposes resources for accessing cached data:

| URI Template | Description |
|--------------|-------------|
| `debate://{debate_id}` | Access debate results by ID |
| `agent://{agent_name}/stats` | Access agent ELO rating and statistics |
| `consensus://{debate_id}` | Access formal verification proofs |
| `trending://topics` | Access current trending topics |

**Example: Reading a Resource**

```python
# From an MCP client
resource = await client.read_resource("debate://mcp_abc12345")
```

## API Reference

### AragoraMCPServer

```python
from aragora.mcp import AragoraMCPServer

server = AragoraMCPServer(
    rate_limits={"run_debate": 10},  # Optional custom limits
    rate_limit_backend="memory",      # "memory" or "redis"
    redis_url="redis://localhost:6379",  # For redis backend
)

await server.run()
```

### Tool Functions

All tool functions are async and return `dict[str, Any]`:

```python
from aragora.mcp.tools import run_debate_tool

result = await run_debate_tool(
    question="What is the best approach?",
    agents="anthropic-api,openai-api",
    rounds=3,
    consensus="majority"
)

if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Answer: {result['final_answer']}")
    print(f"Confidence: {result['confidence']}")
```

### Standalone Usage

Tools can be used without the MCP server:

```python
import asyncio
from aragora.mcp.tools_module.debate import run_debate_tool
from aragora.mcp.tools_module.knowledge import query_knowledge_tool

async def main():
    # Run a debate
    debate = await run_debate_tool(
        question="Should we use GraphQL or REST?",
        agents="anthropic-api,openai-api"
    )

    # Query related knowledge
    knowledge = await query_knowledge_tool(
        query="API design best practices",
        min_confidence=0.7
    )

    print(f"Debate result: {debate['final_answer']}")
    print(f"Related knowledge: {knowledge['count']} nodes found")

asyncio.run(main())
```

## Error Handling

All tools follow a consistent error response pattern:

```python
# Success response
{
    "success": True,
    "data": {...},
    # Additional fields specific to the tool
}

# Error response
{
    "error": "Description of what went wrong"
}

# Rate limit error
{
    "error": "Rate limit exceeded for run_debate. Try again in 30s",
    "rate_limited": True
}
```

## Dependencies

**Required:**
- `mcp` - Model Context Protocol library

**Optional:**
- `redis` - For Redis-backed rate limiting
- `playwright` - For browser automation tools

Install all optional dependencies:

```bash
pip install redis playwright
playwright install
```

## See Also

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [Aragora Debate Engine](../debate/README.md)
- [Knowledge Mound](../knowledge/mound/README.md)
- [Control Plane](../control_plane/README.md)
