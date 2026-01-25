# Aragora Examples

Quick examples to demonstrate Aragora's core capabilities.

## Prerequisites

Set at least one API key:
```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."  # for Grok
```

For best results, set 2-3 keys to enable multi-agent debates.

## Examples

### 1. Simple Multi-Agent Debate (2-5 min)

```bash
python examples/01_simple_debate.py
```

Multiple AI agents debate a topic, critique each other, and reach consensus. This demonstrates the core value proposition of Aragora.

**What you'll see:**
- Agents with different roles (proposer, critic, synthesizer)
- Multi-round debate with critique and revision
- Consensus detection with confidence score

### 2. Tournament & Leaderboard (10-30 min)

```bash
python examples/02_tournament.py
```

Agents compete across multiple topics with ELO ranking. Shows how Aragora tracks agent performance over time.

**What you'll see:**
- Round-robin tournament format
- Multiple debate tasks
- Final standings with wins, points, and ELO ratings

### 3. Nomic Loop - Self-Improvement (5-10 min)

```bash
python examples/03_nomic_loop.py
```

Demonstrates Aragora's unique self-improvement capability (runs in dry-run mode).

**What you'll see:**
- Debate phase: agents propose improvements
- Design phase: agents architect the solution
- Implement phase: code generation (simulated)
- Verify phase: testing (simulated)

### 4. Gauntlet Showcase - AI Decision Assurance (1-2 min)

```bash
python examples/04_gauntlet_showcase.py
```

Demonstrates Aragora's Gauntlet system for adversarial validation and compliance stress-testing. Showcases a real debate from 10 AI models.

**What you'll see:**
- Regulatory personas (GDPR, HIPAA, AI Act, Security)
- Decision Receipt generation (JSON, HTML, PDF)
- Real strategic debate results from 12 AI models
- CLI usage for CI/CD integration

**No API keys required** - views pre-computed results.

### 5. TypeScript SDK Integration (5 min)

```bash
# Start server first
python -m aragora.server.unified_server --port 8080

# In another terminal
npx ts-node examples/05_typescript_sdk.ts
```

Demonstrates using the Aragora TypeScript/JavaScript SDK to integrate debates into your applications.

**What you'll see:**
- Client initialization and health check
- Creating and polling debates programmatically
- Querying agent rankings and debate history
- Proper error handling patterns

**Prerequisites:**
- Node.js 18+
- `npm install aragora-js` (or use from `aragora-js/` directory)

## Troubleshooting

**"Need at least 2 agents"**
- Set more API keys (at least 2 providers needed for debate)

**Timeout errors**
- Some API providers may be slow; wait for completion
- Try setting `ARAGORA_AGENT_TIMEOUT` to increase timeout

**Rate limit errors**
- Set `OPENROUTER_API_KEY` for automatic fallback
- Wait and retry

## SDK Demo Apps

### Python Debate CLI

A full-featured CLI tool demonstrating the Python SDK with auth, tournaments, and onboarding:

```bash
cd examples/python-debate
pip install aragora-client python-dotenv

# Run debates
python main.py debate "Should we use Kubernetes?"
python main.py stream "Design a rate limiter"

# View rankings
python main.py rankings

# Tournaments
python main.py tournament create --name "Q1 Showdown" --agents claude gpt gemini
python main.py tournament list

# Authentication
python main.py auth login --email user@example.com
python main.py auth apikeys list

# Onboarding
python main.py onboarding
```

See [python-debate/README.md](./python-debate/README.md) for details.

### TypeScript Web App

A web app demonstrating the TypeScript SDK with real-time streaming, tabs, and auth:

```bash
cd examples/typescript-web
npm install
npm run dev
# Open http://localhost:5173
```

**Features:**
- Real-time debate streaming via WebSocket
- Tournament creation and management
- Agent rankings leaderboard
- Authentication with token persistence

See [typescript-web/README.md](./typescript-web/README.md) for details.

### Node.js Slack Bot

A Slack bot that enables AI debates directly from Slack channels:

```bash
cd examples/nodejs-slack-bot
npm install
cp .env.example .env
# Edit .env with your Slack credentials
npm run dev
```

**Slack Commands:**
- `/debate <topic>` - Start a multi-agent debate
- `/rankings` - View agent leaderboard
- `/tournament <name>` - Create a tournament

**Features:**
- Real-time debate streaming to threads
- Interactive buttons and modals
- Socket Mode for development

See [nodejs-slack-bot/README.md](./nodejs-slack-bot/README.md) for setup instructions.

## Full Documentation

- [Python SDK Quickstart](../docs/guides/python-quickstart.md)
- [TypeScript SDK Quickstart](../docs/guides/typescript-quickstart.md)
- [CLAUDE.md](../CLAUDE.md) - Architecture overview
- [docs/STATUS.md](../docs/STATUS.md) - Feature status
- [scripts/nomic_loop.py](../scripts/nomic_loop.py) - Full nomic loop implementation
