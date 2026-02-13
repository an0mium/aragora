# Aragora Demo Deployment

One-command deployment of the full Aragora platform in demo/offline mode.

## Quick Start

```bash
docker compose -f deploy/demo/docker-compose.yml up --build
```

Then open:
- **Dashboard:** http://localhost:3000
- **API:** http://localhost:8080/api/v1/health
- **WebSocket:** ws://localhost:8765/ws

## What You Get

- **Backend** (port 8080) — REST API + WebSocket server running in offline mode with SQLite
- **Frontend** (port 3000) — Next.js dashboard showing live debates, agent leaderboard, and analytics
- **Demo Data** — 8 AI agents with ELO ratings, 10 completed debates, trending topics

No API keys, external databases, or Redis required.

## Architecture

```
┌──────────────┐     ┌──────────────┐
│   Frontend   │────>│   Backend    │
│  (Next.js)   │     │  (Python)    │
│  :3000       │     │  :8080 API   │
│              │     │  :8765 WS    │
└──────────────┘     └──────────────┘
                           │
                     ┌─────┴─────┐
                     │  SQLite   │
                     │  (volume) │
                     └───────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_OFFLINE` | `true` | No external API calls |
| `ARAGORA_DEMO_MODE` | `true` | Mock agents, demo data |
| `ARAGORA_SEED_DEMO` | `true` | Auto-seed demo data on startup |
| `ARAGORA_API_PORT` | `8080` | Backend API port |
| `ARAGORA_WS_PORT` | `8765` | WebSocket port |

## Connecting Real API Keys

To use real LLM agents instead of mock agents, add API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
OPENAI_API_KEY=sk-... \
ARAGORA_OFFLINE=false \
ARAGORA_DEMO_MODE=false \
docker compose -f deploy/demo/docker-compose.yml up --build
```

## Troubleshooting

**Frontend shows "connection refused"**: The frontend waits for the backend health check to pass before starting. If the backend takes longer than expected, increase `start_period` in the compose file.

**Port already in use**: Change the host port mappings in the compose file (e.g., `"3001:3000"`).

**Stale data**: Remove the volume to reset: `docker compose -f deploy/demo/docker-compose.yml down -v`
