# Aragora MCP Setup Guide

Aragora exposes 80 tools through the [Model Context Protocol](https://modelcontextprotocol.io/), turning Claude Desktop and Claude Code into a decision intelligence platform. This guide gets you running in under 5 minutes.

## What You Get

Once configured, Claude can:

- **Stress-test decisions** — multi-agent adversarial debates on any question
- **Red-team content** — gauntlet analysis of specs, policies, and code
- **Search and build knowledge** — persistent Knowledge Mound across sessions
- **Run security audits** — 9 audit tools with preset profiles
- **Verify consensus** — formal proofs (Z3/Lean) that reasoning is sound
- **Manage workflows** — DAG-based automation templates
- **Self-improve codebases** — autonomous assessment and improvement cycles
- **Track evidence** — citation chains with provenance

## Quick Start

### 1. Install

```bash
pip install aragora
# Verify:
python -c "from aragora.mcp.server import main; print('OK')"
```

### 2. Set API Keys

At minimum one LLM provider key. More keys = more diverse agents in debates.

```bash
# Required (at least one):
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Recommended (fallback on rate limits):
export OPENROUTER_API_KEY="sk-or-..."

# Optional (additional agents):
export MISTRAL_API_KEY="..."
export GEMINI_API_KEY="..."
```

### 3. Configure Your Client

#### Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "aragora": {
      "command": "python",
      "args": ["-m", "aragora.mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key",
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

#### Claude Code

Add to `.claude/settings.json` or `~/.claude.json`:

```json
{
  "mcpServers": {
    "aragora": {
      "command": "python",
      "args": ["-m", "aragora.mcp.server"]
    }
  }
}
```

API keys are read from your shell environment, so no `env` block needed if they're already exported.

### 4. Verify

Restart Claude Desktop/Code. You should see Aragora's tools available. Test with:

> "Use the list_agents tool to show available debate agents"

## Tool Catalog (80 Tools)

### Core Decision Making (5)

| Tool | Description |
|------|-------------|
| `run_debate` | Multi-agent adversarial debate on any question |
| `run_gauntlet` | Red-team analysis of specs, policies, or code |
| `list_agents` | Show available agents with ELO ratings |
| `get_debate` | Retrieve a completed debate by ID |
| `search_debates` | Search past debates by keyword |

### Agent Intelligence (5)

| Tool | Description |
|------|-------------|
| `get_agent_history` | Agent performance over time |
| `get_consensus_proofs` | Formal proofs for a debate outcome |
| `list_trending_topics` | Current trending topics from Pulse |
| `get_agent_lineage` | Agent evolution/ancestry tree |
| `breed_agents` | Create new agents from successful parents |

### Memory (3)

| Tool | Description |
|------|-------------|
| `query_memory` | Search persistent multi-tier memory |
| `store_memory` | Save insights to memory continuum |
| `get_memory_pressure` | Monitor memory system health |

### Knowledge Mound (6)

| Tool | Description |
|------|-------------|
| `query_knowledge` | Semantic search across knowledge base |
| `store_knowledge` | Persist structured knowledge |
| `get_knowledge_stats` | Knowledge base metrics and health |
| `get_decision_receipt` | Cryptographic receipt for a decision |
| `verify_decision_receipt` | Verify receipt integrity (SHA-256) |
| `build_decision_integrity` | Full integrity report for a decision |

### Verification & Evidence (6)

| Tool | Description |
|------|-------------|
| `verify_consensus` | Formal verification of debate consensus |
| `generate_proof` | Generate Z3/Lean verification proof |
| `verify_plan` | Verify plan against constraints |
| `search_evidence` | Find supporting evidence |
| `cite_evidence` | Create evidence citations with provenance |
| `verify_citation` | Verify citation accuracy |

### Audit & Compliance (9)

| Tool | Description |
|------|-------------|
| `list_audit_presets` | Available audit profiles |
| `list_audit_types` | Supported audit categories |
| `get_audit_preset` | Details of a specific preset |
| `create_audit_session` | Start a new audit session |
| `run_audit` | Execute audit with selected profile |
| `get_audit_status` | Check audit progress |
| `get_audit_findings` | Retrieve audit findings |
| `update_finding_status` | Triage audit findings |
| `run_quick_audit` | One-shot audit (no session setup) |

### Workflow Automation (4)

| Tool | Description |
|------|-------------|
| `run_workflow` | Execute a DAG-based workflow |
| `get_workflow_status` | Check workflow progress |
| `list_workflow_templates` | Available pre-built workflows |
| `cancel_workflow` | Cancel a running workflow |

### Checkpoints (4)

| Tool | Description |
|------|-------------|
| `create_checkpoint` | Save debate state |
| `list_checkpoints` | Available checkpoints |
| `resume_checkpoint` | Resume debate from checkpoint |
| `delete_checkpoint` | Remove a checkpoint |

### Forks (2)

| Tool | Description |
|------|-------------|
| `fork_debate` | Branch a debate to explore alternatives |
| `get_forks` | List forks of a debate |

### Control Plane (11)

| Tool | Description |
|------|-------------|
| `register_agent` | Register an agent in the control plane |
| `unregister_agent` | Remove agent registration |
| `list_registered_agents` | All registered agents |
| `get_agent_health` | Agent health status |
| `submit_task` | Submit task for scheduling |
| `get_task_status` | Check task status |
| `cancel_task` | Cancel a scheduled task |
| `list_pending_tasks` | View task queue |
| `get_control_plane_status` | Overall system status |
| `trigger_health_check` | Force health check |
| `get_resource_utilization` | System resource usage |

### Integrations (4)

| Tool | Description |
|------|-------------|
| `trigger_external_webhook` | Send webhook to external service |
| `list_integrations` | Configured integrations |
| `test_integration` | Test an integration endpoint |
| `get_integration_events` | Recent integration events |

### Canvas (7)

| Tool | Description |
|------|-------------|
| `canvas_create` | Create visual collaboration canvas |
| `canvas_get` | Retrieve canvas state |
| `canvas_add_node` | Add node to canvas |
| `canvas_add_edge` | Connect canvas nodes |
| `canvas_execute_action` | Run action on canvas |
| `canvas_list` | List all canvases |
| `canvas_delete_node` | Remove canvas node |

### Pipeline (4)

| Tool | Description |
|------|-------------|
| `run_pipeline` | Execute idea-to-execution pipeline |
| `extract_goals` | Extract goals from natural language |
| `get_pipeline_status` | Check pipeline progress |
| `advance_pipeline_stage` | Move to next pipeline stage |

### Codebase Analysis (4)

| Tool | Description |
|------|-------------|
| `search_codebase` | AST-based code search |
| `get_symbol` | Get symbol definition and references |
| `get_dependencies` | Dependency graph for a module |
| `get_codebase_structure` | High-level codebase map |

### Self-Improvement (5)

| Tool | Description |
|------|-------------|
| `assess_codebase` | Automated codebase health assessment |
| `generate_improvement_goals` | Prioritized improvement suggestions |
| `run_self_improvement` | Execute improvement cycle |
| `get_daemon_status` | Self-improvement daemon status |
| `trigger_improvement_cycle` | Trigger an improvement cycle |

## Common Workflows

### Decision Stress-Test

> "Run a debate on whether we should migrate to microservices. Use 5 agents and 3 rounds."

This calls `run_debate` → returns structured result with proposals, critiques, consensus score, and a decision receipt.

### Security Audit

> "Run a quick security audit on this API specification: [paste spec]"

Calls `run_quick_audit` → returns findings with severity ratings and remediation suggestions.

### Knowledge-Driven Development

> "Store this architecture decision in the knowledge base, then search for related past decisions."

Calls `store_knowledge` → `query_knowledge` → builds on institutional memory.

### Self-Improving Codebase

> "Assess the codebase health and suggest the top 5 improvements."

Calls `assess_codebase` → `generate_improvement_goals` → optionally `run_self_improvement` with `dry_run=true`.

## Implementation Mode

When running inside the Nomic Loop's implementation phase, set `ARAGORA_MCP_IMPL_MODE=1` to restrict to 27 safe tools (codebase search, knowledge, memory, pipeline, self-improve assessment). This prevents accidental debate or audit execution during autonomous code generation.

```bash
ARAGORA_MCP_IMPL_MODE=1 python -m aragora.mcp.server
```

## Transport Modes

The MCP server supports multiple transports:

| Mode | Command | Use Case |
|------|---------|----------|
| stdio (default) | `python -m aragora.mcp.server` | Claude Desktop, Claude Code |
| SSE | `python -m aragora.mcp.server --transport sse` | Browser-based clients |
| Streamable HTTP | `python -m aragora.mcp.server --transport streamable-http` | Custom integrations |

## Troubleshooting

**Tools not appearing?** Restart Claude Desktop/Code after config changes.

**"No valid agents" error?** Ensure at least one API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) is set.

**Rate limit errors?** Add `OPENROUTER_API_KEY` for automatic fallback.

**Import errors?** Install optional dependencies: `pip install aragora[mcp]`

**Server won't start?** Check: `python -c "from aragora.mcp.server import main; main()"` for error output.

## Further Reading

- [MCP Integration Guide](./MCP_INTEGRATION.md) — detailed tool parameters and resource templates
- [MCP Advanced Guide](./MCP_ADVANCED.md) — SSE transport, rate limiting, caching strategies
- [Model Context Protocol spec](https://modelcontextprotocol.io/) — official MCP documentation
