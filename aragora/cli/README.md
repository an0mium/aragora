# CLI Module

Command-line interface for orchestrating multi-agent debates and managing Aragora.

## Quick Start

```bash
# Run a debate
aragora ask "Design a rate limiter" --agents grok,anthropic-api,openai-api,gemini

# Check system status
aragora status

# Start the server
aragora serve --port 8080
```

## Key Components

| File | Purpose |
|------|---------|
| `main.py` | Entry point, re-exports command functions |
| `parser.py` | Argument parser construction for all subcommands |
| `commands/` | Command implementations organized by category |
| `gt.py` | Self-improvement CLI (Nomic Loop integration) |
| `repl.py` | Interactive debate REPL |
| `gauntlet.py` | Security and code audit runner |

## Architecture

```
cli/
├── main.py              # Entry point, backward-compat re-exports
├── parser.py            # Argument parser (build_parser)
├── commands/            # Command implementations
│   ├── debate.py        # run_debate, cmd_ask, parse_agents
│   ├── stats.py         # cmd_stats, cmd_patterns, cmd_memory, cmd_elo
│   ├── status.py        # cmd_status, cmd_validate_env, cmd_doctor
│   ├── server.py        # cmd_serve
│   ├── tools.py         # cmd_modes, cmd_templates, cmd_improve
│   ├── delegated.py     # Thin wrappers for other CLI modules
│   └── decide.py        # Decision planning commands
├── agents.py            # Agent listing and discovery
├── audit.py             # Codebase audit commands
├── backup.py            # Backup management
├── batch.py             # Batch debate processing
├── bench.py             # Performance benchmarking
├── billing.py           # Usage and cost tracking
├── config.py            # Configuration management
├── demo.py              # Demo/showcase commands
├── doctor.py            # System health diagnostics
├── documents.py         # Document ingestion
├── export.py            # Debate export (JSON, SARIF)
├── gauntlet.py          # Security audit runner
├── gt.py                # Self-improvement (Nomic Loop)
├── init.py              # Project initialization
├── knowledge.py         # Knowledge base management
├── marketplace.py       # Skills marketplace
├── openclaw.py          # OpenClaw gateway CLI
├── replay.py            # Debate replay
├── repl.py              # Interactive REPL
├── review.py            # Code review commands
├── rlm.py               # RLM (Recursive Language Model) CLI
├── security.py          # Security scan commands
├── setup.py             # First-run setup wizard
├── template.py          # Template management
├── tenant.py            # Multi-tenancy management
└── training.py          # Agent training commands
```

## Core Commands

### Debate Commands

```bash
# Basic debate
aragora ask "Your question here"

# With specific agents
aragora ask "Design auth" --agents grok,anthropic-api,openai-api,gemini

# Multi-round debate
aragora ask "Complex topic" --rounds 9 --consensus majority

# With knowledge base context
aragora ask "Based on our docs..." --workspace my-workspace
```

### Status and Diagnostics

```bash
# Environment status
aragora status

# Validate configuration
aragora validate

# Health diagnostics
aragora doctor

# Validate environment variables
aragora validate-env
```

### Server Management

```bash
# Start server
aragora serve --port 8080

# With hot reload
aragora serve --reload

# Production mode
aragora serve --workers 4
```

### Self-Improvement (gt)

```bash
# Run Nomic Loop
aragora gt run --cycles 3

# Staged execution
aragora gt debate
aragora gt design
aragora gt implement
aragora gt verify
```

### Security Audits

```bash
# Run gauntlet
aragora gauntlet run --target ./src

# Codebase audit
aragora audit --path ./src --type security

# Export findings
aragora export --format sarif --output findings.sarif
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_API_URL` | `http://localhost:8080` | API server URL |
| `ARAGORA_DB` | `agora_memory.db` | SQLite database path |
| `ARAGORA_WORKSPACE` | `default` | Default workspace ID |
| `ANTHROPIC_API_KEY` | - | Required for Claude agents |
| `OPENAI_API_KEY` | - | Required for GPT agents |

## Adding New Commands

1. Create a command function in `commands/` or a new module:

```python
# In commands/myfeature.py
def cmd_myfeature(args) -> int:
    """My feature command."""
    print(f"Running myfeature with: {args.param}")
    return 0
```

2. Add parser in `parser.py`:

```python
def _add_myfeature_parser(subparsers):
    p = subparsers.add_parser("myfeature", help="My feature")
    p.add_argument("--param", help="A parameter")
    p.set_defaults(func=cmd_myfeature)
```

3. Call `_add_myfeature_parser(subparsers)` in `build_parser()`.

## Integration Points

| Module | Integration |
|--------|-------------|
| `aragora.debate` | Core debate orchestration via `Arena` |
| `aragora.agents` | Agent discovery and initialization |
| `aragora.memory` | Memory persistence (ContinuumMemory) |
| `aragora.knowledge` | Knowledge base queries |
| `aragora.nomic` | Self-improvement loop |
| `aragora.server` | Embedded server startup |

## Related

- `aragora/modes/` - Debate mode definitions
- `aragora/server/` - HTTP API server
- `aragora/nomic/` - Self-improvement orchestration
- `docs/CLI_REFERENCE.md` - Full command reference
