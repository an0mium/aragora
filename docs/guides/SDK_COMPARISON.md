# SDK Comparison: Python vs TypeScript

Aragora provides official SDKs for the same HTTP API surface (`/api/v1`).

| Feature | Python (`aragora-sdk`) | TypeScript (`@aragora/sdk`) |
|---|---|---|
| Install | `pip install aragora-sdk` | `npm install @aragora/sdk` |
| Clients | Sync + async (`AragoraClient`, `AragoraAsyncClient`) | Async-first |
| Types | Pydantic models + type hints | Full TypeScript types |
| Streaming | WebSocket support | WebSocket + async iterator helpers |

Deprecated: `aragora-client` is a legacy async-only Python client. New integrations should use `aragora-sdk`.

## Minimal Examples

Python (sync):

```python
from aragora_sdk import AragoraClient

client = AragoraClient(base_url="http://localhost:8080", api_key="your-api-key")
resp = client.debates.create(task="Should we adopt microservices?")
print(resp)
```

TypeScript:

```ts
import { createClient } from "@aragora/sdk";

const client = createClient({ baseUrl: "http://localhost:8080", apiKey: "your-api-key" });
const resp = await client.debates.create({ task: "Should we adopt microservices?" });
console.log(resp);
```

## Reference

- Python SDK docs: `sdk/python/README.md`
- TypeScript SDK docs: `sdk/typescript/README.md`
- API spec: `openapi.json`

