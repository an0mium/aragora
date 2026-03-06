# Real Debates Everywhere — Design Document

> Date: 2026-03-06
> Goal: Every debate on aragora.ai uses real LLM models via real APIs, with OpenRouter as universal fallback. Cached results prevent redundant spend.

## Problem

The aragora.ai public site serves canned mock debates instead of real LLM output. Registration is broken. The CLI demo depends on a separate package that fails to import. Users cannot experience real adversarial debate.

## Requirements

1. **No mocks in user-facing paths** — playground, landing, and CLI all run real LLM debates
2. **OpenRouter universal fallback** — every API call falls back to OpenRouter if the primary provider is unavailable or unfunded
3. **Debate caching** — identical prompts with identical model sets return cached results instantly (no re-spend)
4. **Registration works** — users can sign up on aragora.ai
5. **CLI demo uses real models** — `aragora demo` runs a real 1-round debate with budget cap

## Architecture

### Debate Cache (Content-Addressed)

```
POST /api/v1/playground/debate { topic, rounds, agents }
  |
  v
compute cache_key = SHA-256(normalize(topic) + sorted(model_ids) + str(rounds))
  |
  v
lookup cache_key in debate_cache_index table (SQLite)
  |
  +-- HIT --> DebateResultStore.get(debate_id) --> return immediately (skip rate limit)
  |
  +-- MISS --> rate_limit_check --> run_real_debate --> persist --> index by cache_key --> return
```

**SQLite schema** (in existing `debate_results.db`):

```sql
CREATE TABLE IF NOT EXISTS debate_cache_index (
    cache_key TEXT PRIMARY KEY,
    debate_id TEXT NOT NULL,
    topic_normalized TEXT NOT NULL,
    model_ids TEXT NOT NULL,
    rounds INTEGER NOT NULL,
    created_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (debate_id) REFERENCES debate_results(id)
);
CREATE INDEX IF NOT EXISTS idx_cache_created ON debate_cache_index(created_at);
```

**Normalization**:
- Topic: `topic.strip().lower()`, collapse whitespace to single space
- Models: sorted alphabetically, joined with `|`
- Rounds: integer
- Key: `hashlib.sha256(f"{normalized_topic}|{model_str}|{rounds}".encode()).hexdigest()`

**TTL**: Cache entries expire with the debate result (30 days default). `cleanup_expired()` deletes orphaned cache index rows.

**Response**: Cached responses include `"cached": true` and `"cached_at": "<iso timestamp>"` so the frontend can distinguish fresh vs cached (optional, for analytics).

### OpenRouter Universal Fallback

**Current state**: `QuotaFallbackMixin` triggers on 401/429/403 errors AFTER a primary call fails. `_get_available_live_agents()` skips agents whose primary API key is missing.

**New behavior**: When building playground agents, always create them routed through OpenRouter with diverse models:

```python
PLAYGROUND_OPENROUTER_AGENTS = [
    ("analyst",     "anthropic/claude-sonnet-4"),
    ("critic",      "openai/gpt-4o"),
    ("synthesizer", "google/gemini-2.0-flash-001"),
]
```

Each agent is an `OpenRouterAgent` instance with a different model. This means:
- Only OPENROUTER_API_KEY is required (one key, multiple models)
- True model diversity (different architectures catch different blind spots)
- No silent fallback to mocks — if OpenRouter is down, return an honest 503

**Primary API keys still used when available**: If ANTHROPIC_API_KEY is set, use `AnthropicAPIAgent` with OpenRouter fallback (existing `QuotaFallbackMixin`). OpenRouter-only mode is the floor, not the ceiling.

### Registration Fix

**Root cause**: `read_json_body()` relies on `Content-Length` header. Cloudflare HTTP/2 + nginx proxy chain may use chunked transfer encoding where Content-Length is absent or 0.

**Fix**: In `read_json_body()`, when `content_length <= 0`, attempt to read from `rfile` with a reasonable max size (1MB) instead of returning empty dict. Also handle `Transfer-Encoding: chunked`.

### CLI Demo

**Current**: Depends on `aragora-debate` package for `StyledMockAgent`. Fails when not importable.

**New**: `aragora demo` runs a real 1-round debate via OpenRouter with:
- Budget cap: $0.02
- Timeout: 30 seconds
- 3 agents (analyst, critic, synthesizer) via OpenRouter
- Falls back to `--offline` mock mode only if OPENROUTER_API_KEY is not set
- `aragora demo --offline` explicitly runs mock mode

### Playground Handler Changes

**Remove from main path**:
- `_run_inline_mock_debate()` — delete or move to test-only
- `_run_styled_mock_debate()` — delete or move to demo-only path

**New main path**:
```
handle_post("/api/v1/playground/debate"):
    1. Parse topic, rounds, agents from body
    2. Compute cache_key
    3. Check cache → return if hit
    4. Check rate limit (only on cache miss)
    5. Build agents: prefer primary APIs, fallback to OpenRouter
    6. Run real debate via start_playground_debate()
    7. Persist result + index in cache
    8. Return result
```

**`/demo` path stays separate**: The demo page at `/demo` can keep its pre-scripted replay for zero-latency showcase. But `/playground` always runs real debates.

## Files Changed

| File | Change |
|------|--------|
| `aragora/server/handlers/playground.py` | Remove mock fallbacks from main path, add cache lookup, wire OpenRouter agents |
| `aragora/storage/debate_store.py` | Add `debate_cache_index` table, `get_by_cache_key()`, `save_cache_index()` methods |
| `aragora/server/handlers/base.py` | Fix `read_json_body()` to handle missing Content-Length |
| `aragora/cli/demo.py` | Run real debate via OpenRouter instead of requiring `aragora-debate` |
| `aragora/agents/fallback.py` | Ensure `_get_available_live_agents()` always returns agents when OPENROUTER_API_KEY is set |

## Files NOT Changed

- Frontend (no changes needed — PlaygroundDebate already handles the API response shape)
- Demo page (`/demo` keeps its pre-scripted replay)
- Existing fallback mixin (works correctly, just needs agents to be created)

## Testing

- Unit tests for cache key normalization and collision resistance
- Unit tests for cache hit/miss flow in playground handler
- Integration test: same topic returns cached result with `"cached": true`
- Integration test: different topic runs fresh debate
- Unit test for `read_json_body()` with missing Content-Length
- Unit test for CLI demo with and without OPENROUTER_API_KEY

## Budget Protection

- Playground debate budget: $0.05 per debate (existing)
- CLI demo budget: $0.02 per debate (new, lower)
- Rate limit: 1 live debate per 10 minutes per IP (existing, only on cache miss)
- Cache prevents re-spend on identical prompts
- OpenRouter has its own rate limits and billing alerts

## Rollout

1. Deploy backend fixes to EC2 (existing deploy-secure.yml pipeline)
2. No frontend deployment needed
3. Monitor: cache hit rate, debate cost, error rate
4. If OpenRouter is down → 503 with clear message, not silent mock
