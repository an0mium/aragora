# Local Development Setup

Run the Aragora backend and frontend together on your machine.

**Prerequisites:** Docker (recommended), or Python 3.11+ and Node 18+.

---

## Quick Start (Docker -- 2 minutes)

No API keys needed. Uses mock agents and SQLite.

```bash
git clone https://github.com/aragora-ai/aragora.git && cd aragora
docker compose -f docker-compose.quickstart.yml up
```

- Dashboard: [http://localhost:3000](http://localhost:3000)
- API: [http://localhost:8080](http://localhost:8080)
- WebSocket: `ws://localhost:8765/ws`

To use real AI agents, create a `.env` and restart:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
docker compose -f docker-compose.quickstart.yml up --force-recreate
```

### Docker Dev Mode (hot reload)

For active development with source-mounted volumes:

```bash
cp .env.example .env   # set at least one API key
docker compose -f docker-compose.dev.yml up
```

Backend auto-restarts on Python file changes. Frontend uses Next.js fast refresh.

---

## Manual Setup (no Docker)

### 1. Backend

```bash
# From repository root
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set at least one API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Start the server
aragora serve --api-port 8080
```

The API is now at `http://localhost:8080` and WebSocket at `ws://localhost:8765/ws`.

### 2. Frontend

In a second terminal:

```bash
cd aragora/live
npm install

# Create local env from template
cp .env.local.example .env.local
# Defaults are correct for local dev:
#   NEXT_PUBLIC_API_URL=http://localhost:8080
#   NEXT_PUBLIC_WS_URL=ws://localhost:8765/ws

npm run dev
```

Dashboard is now at [http://localhost:3000](http://localhost:3000).

---

## Demo Mode (no API keys)

Run the backend with mock agents and seed data -- no provider keys required:

```bash
aragora serve --demo --api-port 8080
```

Then start the frontend normally (step 2 above). You can explore the full UI,
run mock debates, and browse sample receipts without spending any API credits.

---

## Verify Your Setup

```bash
# Liveness probe
curl http://localhost:8080/healthz

# Full readiness check (DB, Redis, providers)
curl http://localhost:8080/api/v1/readiness | python -m json.tool

# Run your first debate
curl -X POST http://localhost:8080/api/v1/debates \
  -H "Content-Type: application/json" \
  -d '{"question": "Should we use PostgreSQL or SQLite for our MVP?", "rounds": 2}'
```

Or from the CLI:

```bash
aragora ask "Should we use PostgreSQL or SQLite for our MVP?" --rounds 2
```

---

## Common Issues

| Symptom | Fix |
|---------|-----|
| CORS errors in browser | Set `ARAGORA_ALLOWED_ORIGINS=http://localhost:3000` in `.env` or export it before starting the server |
| WebSocket won't connect | Verify `NEXT_PUBLIC_WS_URL=ws://localhost:8765/ws` in `aragora/live/.env.local` |
| "No providers available" | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, or use `--demo` mode |
| Frontend can't reach API | Confirm `NEXT_PUBLIC_API_URL=http://localhost:8080` in `.env.local` and that the backend is running |
| Port 8080 already in use | Use `aragora serve --api-port 9090` and update `NEXT_PUBLIC_API_URL` to match |
| Rate limit errors (429) | Add `OPENROUTER_API_KEY` to `.env` for automatic fallback routing |

### Port Summary

| Service | Default Port | Flag / Env Var |
|---------|-------------|----------------|
| API (HTTP) | 8080 | `--api-port` / `ARAGORA_API_PORT` |
| WebSocket | 8765 | `--ws-port` / `ARAGORA_WS_PORT` |
| Frontend | 3000 | Next.js default (`-p` flag to change) |

---

## Next Steps

- [SME Quickstart](./SME_QUICKSTART.md) -- first debate, playbooks, integrations
- [Environment Reference](../reference/ENVIRONMENT.md) -- all config variables
- [API Reference](../api/API_REFERENCE.md) -- REST endpoints
- [Enterprise Features](../enterprise/ENTERPRISE_FEATURES.md) -- SSO, RBAC, compliance
