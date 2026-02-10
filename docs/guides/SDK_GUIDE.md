# Aragora SDKs

Aragora ships multiple SDK packages. For remote API integrations, use:

- Python: `aragora-sdk` (import `aragora_sdk`)
- TypeScript: `@aragora/sdk`

The full control plane package (`aragora`) is intended for self-hosting and local development (server + CLI).

Deprecated: `aragora-client` (import `aragora_client`) is a legacy async-only client kept for compatibility. New integrations should migrate to `aragora-sdk`.

Prefer `/api/v1` endpoints for SDK usage. Unversioned `/api/...` endpoints remain supported but are considered legacy for SDK clients.

## Installation

```bash
pip install aragora-sdk
npm install @aragora/sdk
```

## Python Quickstart

Synchronous:

```python
from aragora_sdk import AragoraClient

client = AragoraClient(base_url="http://localhost:8080", api_key="your-api-key")
debate = client.debates.create(task="Should we adopt microservices?")
print(debate["debate_id"])
```

Async:

```python
import asyncio
from aragora_sdk import AragoraAsyncClient

async def main() -> None:
    async with AragoraAsyncClient(
        base_url="http://localhost:8080",
        api_key="your-api-key",
    ) as client:
        debate = await client.debates.create(task="Should we adopt microservices?")
        print(debate["debate_id"])

asyncio.run(main())
```

## TypeScript Quickstart

```ts
import { createClient } from "@aragora/sdk";

const client = createClient({
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key",
});

const debate = await client.debates.create({ task: "Should we adopt microservices?" });
console.log(debate.debate_id);
```

## Reference

- Python SDK docs: `sdk/python/README.md`
- TypeScript SDK docs: `sdk/typescript/README.md`
- API spec: `openapi.json`

