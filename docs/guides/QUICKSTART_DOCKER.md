# Quickstart: Docker (Zero Install)

Run Aragora without installing Python, dependencies, or anything except Docker.

## Option A: With Your API Keys (Recommended)

Run a real multi-agent debate server in one command:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here

docker compose -f docker-compose.simple.yml up -d
```

The server starts on `http://localhost:8080`. Run a debate:

```bash
curl -X POST http://localhost:8080/api/v1/debates \
  -H "Content-Type: application/json" \
  -d '{"task": "Should we use microservices or a monolith?", "rounds": 3}'
```

Health check: `http://localhost:8080/healthz`

To stop: `docker compose -f docker-compose.simple.yml down`

## Option B: Demo Mode (No API Keys)

Try the full platform without any API keys -- uses mock agents and pre-loaded demo data:

```bash
docker compose -f deploy/demo/docker-compose.yml up --build
```

This starts:
- **Backend API** on `http://localhost:8080` (REST + health endpoints)
- **Frontend UI** on `http://localhost:3000` (Next.js dashboard)
- **WebSocket** on `ws://localhost:8765` (live debate streaming)

No external database, no API keys, no configuration. SQLite and mock agents handle everything.

## Option C: Self-Hosted with PostgreSQL

For persistent production deployments:

```bash
docker compose up -d
```

This uses the default `docker-compose.yml` which includes PostgreSQL, Redis, and the full service stack.

## What You Can Do

| Action | Command |
|--------|---------|
| Run a debate | `POST /api/v1/debates` with `{"task": "...", "rounds": 3}` |
| Get results | `GET /api/v1/debates/{id}` |
| Stream live | Connect WebSocket to `ws://localhost:8765/ws` |
| Health check | `GET /healthz` |
| List agents | `GET /api/v1/agents` |
| View receipt | `GET /api/v1/receipts/{id}` |

## Next Steps

| Goal | Guide |
|------|-------|
| Run locally without Docker | [Quickstart (Python)](QUICKSTART.md) |
| AI code review on PRs | [Developer Quickstart](../QUICKSTART_DEVELOPER.md) |
| Kubernetes deployment | See `deploy/kubernetes/` |
| Full API reference | [API Reference](../api/API_REFERENCE.md) |
