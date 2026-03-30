# Quickstart

Get from zero to a working Aragora debate in under a minute.

This guide is aligned with the current product surface:

- `aragora quickstart` is the fastest CLI path from a question to a saved receipt.
- `aragora-debate` is the standalone Python package for embedding the debate engine directly.
- `deploy/demo/docker-compose.yml` brings up the offline demo stack with the web UI and API.

## 1. Fastest Path: CLI Quickstart

Install the CLI package:

```bash
pip install aragora
```

Run the zero-config demo path:

```bash
aragora quickstart --demo --no-browser
```

What happens:

- Aragora runs a short demo debate with local mock agents
- It saves a receipt artifact under `.aragora/receipts/`
- It reports whether the run was `demo` or `live`

If you want structured stdout for scripting or CI:

```bash
aragora quickstart --demo --no-browser --json
```

For the full CLI walkthrough, saved artifact behavior, and flags, see [QUICKSTART_CLI.md](QUICKSTART_CLI.md).

## 2. Live Quickstart With a Real Provider

Export a supported API key, then run a one-question debate:

```bash
export OPENAI_API_KEY=sk-...
aragora quickstart --question "Should we adopt GraphQL for our mobile API?" --no-browser
```

Quickstart auto-detects supported providers from your environment and falls back to demo mode if none are available.

You can also provide a key inline for a first run:

```bash
aragora quickstart \
  --provider openai \
  --api-key sk-... \
  --save-key \
  --question "Should we ship this change?" \
  --no-browser
```

## 3. Standalone Python Package

If you want the debate engine without the broader CLI or server surface, use `aragora-debate`:

```bash
pip install aragora-debate
```

Offline example with styled mock agents:

```python
import asyncio
from aragora_debate import Arena, DebateConfig, StyledMockAgent


async def main() -> None:
    agents = [
        StyledMockAgent("analyst", style="supportive"),
        StyledMockAgent("critic", style="critical"),
        StyledMockAgent("pm", style="balanced"),
    ]

    result = await Arena(
        question="Should we migrate to microservices?",
        agents=agents,
        config=DebateConfig(rounds=2),
    ).run()

    print(result.receipt.to_markdown())


asyncio.run(main())
```

For more standalone package examples, see [`aragora-debate/README.md`](../aragora-debate/README.md).

## 4. Self-Host the Full Demo Stack

Bring up the offline demo stack with the backend, WebSocket server, and frontend UI:

```bash
docker compose -f deploy/demo/docker-compose.yml up --build
```

Then visit:

- Landing page: http://localhost:3000
- Public proof demo: http://localhost:3000/demo
- Question-entry flow: http://localhost:3000/try
- Standalone playground: http://localhost:3000/playground
- API docs (Swagger): http://localhost:8080/api/v1/docs
- API docs (ReDoc): http://localhost:8080/api/v1/redoc

The demo stack runs in offline mode with mock agents and SQLite, so it does not require external provider credentials.

## 5. Next Steps

| Guide | What you'll learn |
|-------|-------------------|
| [Quickstart CLI](QUICKSTART_CLI.md) | Current CLI-first onboarding path and flags |
| [Developer Quickstart](QUICKSTART_DEVELOPER.md) | Local development workflow |
| [API Reference](api/API_REFERENCE.md) | REST API endpoints and models |
| [Start Here](START_HERE.md) | Product overview and architecture |
