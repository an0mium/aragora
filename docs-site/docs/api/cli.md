---
title: Aragora CLI Reference
description: Aragora CLI Reference
---

# Aragora CLI Reference

Complete command-line interface reference for Aragora - Control Plane for Multi-Agent Deliberation.

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

Note: `--db` controls the local CritiqueStore file. To keep runtime data out of the
repo root, set `ARAGORA_DATA_DIR` and pass `--db "$ARAGORA_DATA_DIR/agora_memory.db"`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_API_URL` | API server URL | `http://localhost:8080` |
| `ANTHROPIC_API_KEY` | Anthropic API key for `anthropic-api` | - |
| `OPENAI_API_KEY` | OpenAI API key for `openai-api` | - |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `XAI_API_KEY` | xAI Grok API key | - |
| `MISTRAL_API_KEY` | Mistral API key (`mistral-api`, `codestral`) | - |
| `OPENROUTER_API_KEY` | OpenRouter key (OpenRouter agents) | - |
| `KIMI_API_KEY` | Moonshot/Kimi API key | - |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default Ollama model | `llama3.2` |
| `DEEPSEEK_API_KEY` | DeepSeek CLI key (`deepseek-cli`) | - |

See [ENVIRONMENT](../getting-started/environment) for the full configuration reference.

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
| `-a, --agents` | `codex,claude` | Comma-separated agents. `codex,claude` are CLI agents; use `anthropic-api,openai-api` if you only have API keys |
| `-r, --rounds` | `3` | Number of debate rounds |
| `-c, --consensus` | `majority` | Consensus mechanism: `majority`, `unanimous`, `judge`, `none` |
| `--context` | - | Additional context for the task |
| `--no-learn` | - | Don't store patterns in memory |
| `--demo` | - | Run with demo agents (no API keys required) |

**Agent Spec Formats:**

| Format | Example | Description |
|--------|---------|-------------|
| `provider` | `anthropic-api` | Just the provider, role assigned by position |
| `provider:role` | `claude:critic` | Provider with explicit role |
| `provider\|model\|persona\|role` | `anthropic-api\|claude-opus\|philosopher\|proposer` | Full pipe format |

Valid roles: `proposer`, `critic`, `synthesizer`, `judge`

**Examples:**

```bash
# Basic debate with default agents
aragora ask "Should we use microservices or monolith?"

# Recommended default if you only have API keys configured
aragora ask "Should we use microservices or monolith?" --agents anthropic-api,openai-api

# Specify agents with explicit roles (colon format)
aragora ask "Design an auth system" -a "anthropic-api:proposer,openai-api:critic,gemini:synthesizer"

# Full pipe format with model and persona
aragora ask "Design an auth system" -a "anthropic-api|claude-opus|philosopher|proposer,openai-api|||critic"

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

**Exposed Tools (current set):**
- Core: `run_debate`, `run_gauntlet`, `list_agents`, `get_debate`, `search_debates`
- Agent stats: `get_agent_history`, `get_consensus_proofs`, `list_trending_topics`
- Memory: `query_memory`, `store_memory`, `get_memory_pressure`
- Forks: `fork_debate`, `get_forks`
- Genesis: `get_agent_lineage`, `breed_agents`
- Checkpoints: `create_checkpoint`, `list_checkpoints`, `resume_checkpoint`, `delete_checkpoint`
- Verification: `verify_consensus`, `generate_proof`
- Evidence: `search_evidence`, `cite_evidence`, `verify_citation`

See `aragora/mcp/tools.py` for the authoritative list and parameter schemas.

---

### knowledge - Knowledge Base Operations

Query, search, and manage the knowledge base.

```bash
aragora knowledge query "What are the payment terms?"
aragora knowledge facts --workspace default
aragora knowledge search "contract expiration"
aragora knowledge process document.pdf
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `query` | Ask a question about the knowledge base |
| `facts` | List, show, or verify facts |
| `search` | Search document chunks using semantic similarity |
| `process` | Process and ingest documents |
| `jobs` | List processing jobs |

**query Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-w, --workspace` | `default` | Workspace ID |
| `--debate` | - | Use multi-agent debate for synthesis |
| `-n, --limit` | `5` | Max facts to include |
| `--json` | - | Output as JSON |

**facts Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `action` | `list` | Action: `list`, `show`, `verify` |
| `-w, --workspace` | `default` | Workspace ID |
| `-t, --topic` | - | Filter by topic |
| `-s, --status` | - | Filter by status: `unverified`, `contested`, `majority_agreed`, `byzantine_agreed`, `formally_proven` |
| `--min-confidence` | `0.0` | Minimum confidence (0-1) |

---

### document-audit - Document Auditing

Audit documents using multi-agent analysis.

```bash
aragora document-audit upload --input ./docs/
aragora document-audit scan --input ./docs/ --type security
aragora document-audit status --session abc123
aragora document-audit report --session abc123 --output report.json
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `upload` | Upload documents for processing |
| `scan` | Scan documents for issues |
| `status` | Check audit session status |
| `report` | Generate audit report |

**upload Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | - | Path to file or directory |
| `-r, --recursive` | - | Process directories recursively |
| `--chunking` | `auto` | Chunking strategy |
| `--chunk-size` | - | Chunk size in tokens |
| `--chunk-overlap` | - | Overlap between chunks |

**scan Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | - | Path to file or directory |
| `-t, --type` | - | Scan type: `security`, `compliance`, `quality` |

---

### documents - Document Management

Upload, list, and manage documents for auditing and analysis.

```bash
aragora documents upload ./files/*.pdf
aragora documents upload ./folder/ --recursive
aragora documents list
aragora documents show doc-123
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `upload` | Upload files or folders for processing |
| `list` | List uploaded documents |
| `show` | Show document details |

**upload Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --recursive` | - | Recursively upload folder contents |
| `--max-depth` | `10` | Maximum folder depth for recursive uploads (-1 for unlimited) |
| `--exclude` | - | Exclude patterns (gitignore-style, can be repeated) |
| `--include` | - | Include only files matching patterns (can be repeated) |
| `--max-size` | `500mb` | Maximum total upload size (e.g., `500mb`, `1gb`) |
| `--max-file-size` | `100mb` | Maximum size per file |
| `--max-files` | `1000` | Maximum number of files to upload |
| `--agent-filter` | - | Use AI agent to filter files by relevance |
| `--filter-prompt` | - | Custom prompt for agent-based filtering |
| `--filter-model` | `gemini-2.0-flash` | Model to use for agent filtering |
| `--dry-run` | - | Show what would be uploaded without uploading |
| `--config` | - | Path to YAML config file for upload settings |
| `--follow-symlinks` | - | Follow symbolic links (default: skip them) |
| `--json` | - | Output results as JSON |

**list Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --limit` | `50` | Maximum documents to show |
| `--json` | - | Output as JSON |

**show Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `doc_id` | - | Document ID (required) |
| `--chunks` | - | Show document chunks |
| `--json` | - | Output as JSON |

**Examples:**

```bash
# Upload a single file
aragora documents upload contract.pdf

# Upload multiple files via glob
aragora documents upload ./contracts/*.pdf

# Recursively upload a folder with exclusions
aragora documents upload ./project/ -r --exclude "*.log" --exclude "node_modules"

# Dry run to preview what would be uploaded
aragora documents upload ./data/ -r --dry-run

# Upload with AI-powered relevance filtering
aragora documents upload ./mixed-docs/ -r --agent-filter --filter-prompt "Include only financial documents"

# List all uploaded documents
aragora documents list --limit 100

# Show document details with chunks
aragora documents show doc-abc123 --chunks --json
```

---

### publish - Generate Shareable Reports

Generate shareable, interactive reports from debate traces.

```bash
aragora publish <debate-id> --format html --output ./reports/
aragora publish latest --format md
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `debate_id` | Debate ID or `latest` for most recent |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format` | `html` | Output format: `html`, `md`, `json` |
| `-o, --output` | `.` | Output directory |

---

### training - Model Training Operations

Export training data and manage fine-tuning jobs for models trained on Aragora debate data.

```bash
aragora training export-sft -o training_data.jsonl
aragora training export-dpo -o dpo_data.jsonl
aragora training train-sft --output-dir ./models/
aragora training list-models
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `export-sft` | Export SFT (Supervised Fine-Tuning) training data |
| `export-dpo` | Export DPO (Direct Preference Optimization) data |
| `export-gauntlet` | Export gauntlet runs as training data |
| `export-all` | Export all training data types |
| `train-sft` | Run SFT training job |
| `train-dpo` | Run DPO training job |
| `train-combined` | Run combined SFT+DPO training |
| `list-models` | List available fine-tuned models |
| `sample` | Sample from training data |
| `stats` | Show training data statistics |
| `test-connection` | Test training infrastructure connection |

**export-sft Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `sft_training_data.jsonl` | Output file path |
| `--min-confidence` | `0.7` | Minimum debate confidence |
| `--min-success-rate` | `0.6` | Minimum pattern success rate |
| `--limit` | `1000` | Maximum records to export |
| `--db-path` | `agora_memory.db` | Database path |

---

## Agent Types

Available agent types for the `--agents` option. The full catalog and defaults live in [AGENTS.md](../core-concepts/agents).

### Direct API agents

| Agent | Description | API Key Required |
|-------|-------------|------------------|
| `anthropic-api` | Claude via Anthropic API | `ANTHROPIC_API_KEY` |
| `openai-api` | OpenAI via API | `OPENAI_API_KEY` |
| `gemini` | Google Gemini | `GEMINI_API_KEY` |
| `grok` | xAI Grok | `XAI_API_KEY` |
| `mistral-api` | Mistral direct API | `MISTRAL_API_KEY` |
| `codestral` | Mistral code model | `MISTRAL_API_KEY` |
| `ollama` | Local Ollama models | `OLLAMA_HOST` |
| `kimi` | Moonshot/Kimi | `KIMI_API_KEY` |
| `demo` | Demo agent (no API required) | - |

### OpenRouter agents

| Agent | Model | API Key Required |
|-------|-------|------------------|
| `deepseek` | DeepSeek V3 (chat) | `OPENROUTER_API_KEY` |
| `deepseek-r1` | DeepSeek R1 (reasoning) | `OPENROUTER_API_KEY` |
| `llama` | Llama 3.3 70B | `OPENROUTER_API_KEY` |
| `mistral` | Mistral Large | `OPENROUTER_API_KEY` |
| `qwen` | Qwen 2.5 Coder | `OPENROUTER_API_KEY` |
| `qwen-max` | Qwen Max | `OPENROUTER_API_KEY` |
| `yi` | Yi Large | `OPENROUTER_API_KEY` |

### CLI agents (local binaries required)

| Agent | CLI Tool | Notes |
|-------|----------|-------|
| `claude` | `claude` | Anthropic Claude CLI (claude-code) |
| `codex` | `codex` | OpenAI Codex CLI |
| `openai` | `openai` | OpenAI CLI |
| `gemini-cli` | `gemini` | Google Gemini CLI |
| `grok-cli` | `grok` | xAI Grok CLI |
| `qwen-cli` | `qwen` | Qwen CLI |
| `deepseek-cli` | `deepseek` | DeepSeek CLI |
| `kilocode` | `kilocode` | KiloCode CLI |

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

- [Getting Started Guide](../getting-started/overview)
- [Agent Selection Guide](../core-concepts/agent-selection)
- [MCP Integration](../guides/mcp-integration)
- [API Reference](./reference)
