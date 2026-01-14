# ADR-005: Composition Over Inheritance for APIs

## Status
Accepted

## Context

The Aragora client SDK needs to expose many API endpoints:
- Debates (CRUD, run, compare, batch)
- Agents (list, get, configure)
- Memory (analytics, snapshots, tiers)
- Gauntlet (run, receipts)
- Replays (list, export, delete)
- And more...

Approaches considered:
1. **Single large class**: All methods on AragoraClient
2. **Inheritance hierarchy**: DebateClient extends BaseClient
3. **Composition**: Separate API classes composed into client

## Decision

Use composition with domain-specific API classes:

```python
class AragoraClient:
    def __init__(self, base_url: str, ...):
        self._session = None
        self._rate_limiter = RateLimiter(rps)

        # Compose domain-specific APIs
        self.debates = DebatesAPI(self)
        self.agents = AgentsAPI(self)
        self.memory = MemoryAPI(self)
        self.gauntlet = GauntletAPI(self)
        self.replays = ReplayAPI(self)
        self.leaderboard = LeaderboardAPI(self)
        self.verification = VerificationAPI(self)
        self.graph_debates = GraphDebatesAPI(self)
        self.matrix_debates = MatrixDebatesAPI(self)
```

Each API class:
```python
class DebatesAPI:
    def __init__(self, client: "AragoraClient"):
        self._client = client

    def create(self, task: str, ...) -> Debate:
        return self._client._post("/api/debates", {...})

    async def create_async(self, task: str, ...) -> Debate:
        return await self._client._post_async("/api/debates", {...})
```

Usage:
```python
client = AragoraClient(base_url="http://localhost:8080")
debate = client.debates.create(task="Should we use microservices?")
agents = client.agents.list()
```

## Consequences

### Positive
- **Discoverability**: `client.debates.<tab>` shows debate methods
- **Organization**: Related methods grouped together
- **Testability**: API classes can be mocked independently
- **Maintainability**: Changes to one domain don't affect others
- **IDE support**: Better autocomplete and documentation

### Negative
- **File size**: Single file is 1,700+ lines (though well-organized)
- **Indirection**: Extra layer between user and HTTP calls
- **Boilerplate**: Each API class needs similar patterns

### Trade-offs Made
- Kept all API classes in one file for simpler imports
- Could extract to separate files if growth continues
- Sync and async variants for each method (duplication but clear API)

## Related
- `aragora/client/client.py` - Main client implementation
- `aragora/client/models.py` - Request/response models
- `aragora/client/websocket.py` - WebSocket handling (separate concern)
