# Aragora CLI Reference

Complete command-line interface reference for Aragora - AI Red Team for Decision Stress-Testing.

## Installation

```bash
pip install aragora
```

## Global Options

```bash
aragora [--version] [--db PATH] [--verbose] <command> [options]
```

| Option | Description |
|--------|-------------|
| `-V, --version` | Show version number |
| `--db PATH` | SQLite database path (default: `agora_memory.db`) |
| `-v, --verbose` | Enable verbose output |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_API_URL` | API server URL | `http://localhost:8080` |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude agents | - |
| `OPENAI_API_KEY` | OpenAI API key for GPT agents | - |
| `OPENROUTER_API_KEY` | OpenRouter API key (fallback) | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `XAI_API_KEY` | xAI Grok API key | - |

---

## Commands

### ask - Run Debates

Run a decision stress-test (debate engine) on a task or question.

```bash
aragora ask "Design a rate limiter" --agents anthropic-api,openai-api --rounds 3
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `task` | The task or question to debate |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-a, --agents` | `codex,claude` | Comma-separated agents. Use `agent:role` for specific roles |
| `-r, --rounds` | `3` | Number of debate rounds |
| `-c, --consensus` | `majority` | Consensus mechanism: `majority`, `unanimous`, `judge`, `none` |
| `--context` | - | Additional context for the task |
| `--no-learn` | - | Don't store patterns in memory |
| `--demo` | - | Run with demo agents (no API keys required) |

**Examples:**

```bash
# Basic debate with default agents
aragora ask "Should we use microservices or monolith?"

# Specify agents and roles
aragora ask "Design an auth system" -a "anthropic-api:proposer,openai-api:critic,gemini:synthesizer"

# Quick demo without API keys
aragora ask "Rate limiter design" --demo
```

---

### gauntlet - Adversarial Stress-Testing

Run comprehensive adversarial stress-testing on documents (specs, architecture, policies, code).

```bash
aragora gauntlet spec.md --input-type spec --profile thorough
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Path to input file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --input-type` | `spec` | Type: `spec`, `architecture`, `policy`, `code`, `strategy`, `contract` |
| `-a, --agents` | `anthropic-api,openai-api` | Agents for stress-testing |
| `-p, --profile` | `default` | Profile: `default`, `quick`, `thorough`, `code`, `policy`, `gdpr`, `hipaa`, `ai_act`, `security`, `sox` |
| `--persona` | - | Regulatory persona for compliance testing |
| `-r, --rounds` | varies | Number of deep audit rounds |
| `--timeout` | varies | Maximum duration in seconds |
| `-o, --output` | - | Output path for Decision Receipt |
| `-f, --format` | `html` | Output format: `json`, `md`, `html` |
| `--verify` | - | Enable formal verification (Z3/Lean) |
| `--no-redteam` | - | Disable red-team attacks |
| `--no-probing` | - | Disable capability probing |
| `--no-audit` | - | Disable deep audit |

**Examples:**

```bash
# Quick spec review
aragora gauntlet api-spec.md -p quick

# Thorough architecture audit
aragora gauntlet architecture.md -t architecture -p thorough -o receipt.html

# GDPR compliance check
aragora gauntlet privacy-policy.md -t policy --persona gdpr

# Security code review with formal verification
aragora gauntlet auth.py -t code -p security --verify
```

---

### review - AI Code Review

Run multi-agent AI code review on a diff or PR.

```bash
aragora review https://github.com/owner/repo/pull/123
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `pr_url` | GitHub PR URL (optional if using `--diff-file` or stdin) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--diff-file` | - | Path to diff file (alternative to PR URL) |
| `--agents` | `anthropic-api,openai-api` | Agents for review |
| `--rounds` | `2` | Number of debate rounds |
| `--focus` | `security,performance,quality` | Focus areas |
| `--output-format` | `github` | Format: `github`, `json`, `html` |
| `--output-dir` | - | Directory to save output artifacts |
| `--demo` | - | Demo mode (no API keys required) |
| `--share` | - | Generate a shareable link |

**Examples:**

```bash
# Review a GitHub PR
aragora review https://github.com/myorg/myrepo/pull/42

# Review a local diff file
aragora review --diff-file changes.diff

# Review from stdin
git diff main | aragora review

# Security-focused review
aragora review PR_URL --focus security --rounds 3
```

---

### serve - Live Debate Server

Run the live debate server for real-time streaming and audience participation.

```bash
aragora serve --api-port 8080 --ws-port 8765
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--api-port` | `8080` | HTTP API port |
| `--ws-port` | `8765` | WebSocket port |
| `--host` | `localhost` | Host to bind to |

---

### batch - Batch Processing

Process multiple debates from a JSONL or JSON file.

```bash
aragora batch debates.jsonl --output results.json
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Path to JSONL or JSON file with debate items |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --server` | - | Submit to server batch API instead of local processing |
| `-u, --url` | `http://localhost:8080` | Server URL |
| `-t, --token` | - | API authentication token |
| `-w, --webhook` | - | Webhook URL for completion notification |
| `--wait` | - | Wait for batch completion (server mode) |
| `-a, --agents` | `anthropic-api,openai-api` | Default agents |
| `-r, --rounds` | `3` | Default rounds |
| `-o, --output` | - | Output path for results JSON |

**Input File Format:**

```jsonl
{"question": "Design a rate limiter", "agents": "anthropic-api,openai-api"}
{"question": "Implement caching", "rounds": 4}
{"question": "Security review", "priority": 10}
```

---

### repl - Interactive Mode

Start an interactive debate session.

```bash
aragora repl --agents anthropic-api,openai-api
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-a, --agents` | `anthropic-api,openai-api` | Agents for debates |
| `-r, --rounds` | `3` | Debate rounds |

---

### init - Initialize Project

Initialize an Aragora project in a directory.

```bash
aragora init ./my-project
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `directory` | Target directory (default: current) |

**Options:**

| Option | Description |
|--------|-------------|
| `-f, --force` | Overwrite existing files |
| `--no-git` | Don't modify `.gitignore` |

---

### config - Manage Configuration

Manage Aragora configuration settings.

```bash
aragora config show
aragora config set api_url http://localhost:8080
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `action` | Action: `show`, `get`, `set`, `env`, `path` |
| `key` | Config key (for `get`/`set`) |
| `value` | Config value (for `set`) |

---

### stats - Memory Statistics

Show memory and learning statistics.

```bash
aragora stats
```

---

### status - Environment Health

Show environment health and agent availability.

```bash
aragora status --server http://localhost:8080
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --server` | `http://localhost:8080` | Server URL to check |

---

### agents - List Agents

List available agents and their configuration status.

```bash
aragora agents --verbose
```

**Options:**

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed descriptions |

---

### patterns - Learned Patterns

Show patterns learned from previous debates.

```bash
aragora patterns --type security --limit 20
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --type` | - | Filter by issue type |
| `--min-success` | `1` | Minimum success count |
| `-l, --limit` | `10` | Max patterns to show |

---

### demo - Quick Demo

Run a quick demo debate without API keys.

```bash
aragora demo rate-limiter
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Demo name: `rate-limiter`, `auth`, `cache` |

---

### templates - Debate Templates

List available debate templates.

```bash
aragora templates
```

---

### export - Export Artifacts

Export debate artifacts to various formats.

```bash
aragora export --debate-id abc123 --format html --output ./exports
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --debate-id` | - | Debate ID to export |
| `-f, --format` | `html` | Format: `html`, `json`, `md` |
| `-o, --output` | `.` | Output directory |
| `--demo` | - | Generate a demo export |

---

### replay - Replay Debates

Replay stored debates from session recordings.

```bash
aragora replay list
aragora replay show abc123
aragora replay play abc123 --speed 2.0
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `action` | Action: `list`, `show`, `play` |
| `id` | Replay ID (for `show`/`play`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-d, --directory` | - | Replays directory |
| `-n, --limit` | `10` | Max replays to list |
| `-s, --speed` | `1.0` | Playback speed |

---

### bench - Benchmark Agents

Benchmark agent performance on standardized tasks.

```bash
aragora bench --agents anthropic-api,openai-api --iterations 5
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-a, --agents` | `anthropic-api,openai-api` | Agents to benchmark |
| `-n, --iterations` | `3` | Iterations per task |
| `-t, --task` | - | Custom benchmark task |
| `-q, --quick` | - | Quick mode (1 iteration) |

---

### doctor - Health Checks

Run system health checks and diagnostics.

```bash
aragora doctor --validate
```

**Options:**

| Option | Description |
|--------|-------------|
| `-v, --validate` | Validate API keys by making test calls |

---

### validate - Validate API Keys

Validate API keys by making test API calls.

```bash
aragora validate
```

---

### improve - Self-Improvement Mode

Run self-improvement analysis on a codebase.

```bash
aragora improve --path ./my-project --focus performance
```

**Options:**

| Option | Description |
|--------|-------------|
| `-p, --path` | Path to codebase (default: current dir) |
| `-f, --focus` | Focus area for improvements |
| `-a, --analyze` | Analyze codebase structure |

---

### badge - Generate Badges

Generate Aragora badges for your README.

```bash
aragora badge --type reviewed --style flat-square
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --type` | `reviewed` | Type: `reviewed`, `consensus`, `gauntlet` |
| `-s, --style` | `flat` | Style: `flat`, `flat-square`, `for-the-badge`, `plastic` |
| `-r, --repo` | - | Link to specific repo |

---

### mcp-server - MCP Server

Run the MCP (Model Context Protocol) server for Claude integration.

```bash
aragora mcp-server
```

Configure in `claude_desktop_config.json`:

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

**Exposed Tools:**
- `run_debate`: Run decision stress-tests
- `run_gauntlet`: Stress-test documents
- `list_agents`: List available agents
- `get_debate`: Retrieve debate results

---

## Agent Types

Available agent types for the `--agents` option:

| Agent | Description | API Key Required |
|-------|-------------|------------------|
| `anthropic-api` | Claude via Anthropic API | `ANTHROPIC_API_KEY` |
| `openai-api` | GPT via OpenAI API | `OPENAI_API_KEY` |
| `gemini` | Google Gemini | `GEMINI_API_KEY` |
| `grok` | xAI Grok | `XAI_API_KEY` |
| `mistral` | Mistral AI | `MISTRAL_API_KEY` |
| `codex` | CLI Claude (requires `claude` CLI) | - |
| `claude` | CLI Claude (alias for codex) | - |
| `demo` | Demo agent (no API required) | - |

**Agent Roles:**

- `proposer` - Generates initial proposals
- `critic` - Critiques and finds weaknesses
- `synthesizer` - Synthesizes consensus

**Role Assignment:**

```bash
# Explicit roles
aragora ask "Design auth" -a "anthropic-api:proposer,openai-api:critic,gemini:synthesizer"

# Auto-assigned roles (first=proposer, last=synthesizer, middle=critic)
aragora ask "Design auth" -a "anthropic-api,openai-api,gemini"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | API key missing |
| `4` | Network error |

---

## Quick Start Examples

```bash
# Run your first debate
aragora demo rate-limiter

# Run a real debate with API keys configured
aragora ask "Design a microservices architecture for e-commerce"

# Stress-test a specification
aragora gauntlet api-spec.md -t spec -p thorough

# Review a PR
aragora review https://github.com/org/repo/pull/123

# Start the live server
aragora serve

# Check system health
aragora doctor --validate
```

---

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [Agent Selection Guide](AGENT_SELECTION.md)
- [MCP Integration](MCP_INTEGRATION.md)
- [API Reference](API_REFERENCE.md)
