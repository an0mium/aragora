# Environment Variable Reference

Complete reference for all environment variables used by Aragora.

## Quick Start

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

## Production Required Variables

These variables **MUST** be set in production (`ARAGORA_ENV=production`). The application will fail loudly if they are missing, preventing silent fallback to localhost defaults.

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_OAUTH_CLIENT_ID` | Google OAuth client ID | `1234567890-abc.apps.googleusercontent.com` |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth client secret | `your-client-secret` |
| `GOOGLE_OAUTH_REDIRECT_URI` | OAuth callback URL | `https://api.aragora.ai/api/auth/oauth/google/callback` |
| `OAUTH_SUCCESS_URL` | Post-login redirect | `https://aragora.ai/auth/success` |
| `OAUTH_ERROR_URL` | Auth error page | `https://aragora.ai/auth/error` |
| `OAUTH_ALLOWED_REDIRECT_HOSTS` | Comma-separated allowed hosts | `aragora.ai,api.aragora.ai` |
| `NEXT_PUBLIC_API_URL` | Frontend API base URL | `https://api.aragora.ai` |
| `NEXT_PUBLIC_WS_URL` | Frontend WebSocket URL | `wss://api.aragora.ai` |

**Warning Behavior:**
- In development mode, missing URLs will trigger console warnings but fall back to `localhost`
- In production mode (`ARAGORA_ENV=production`), missing OAuth URLs will cause startup failures
- Frontend components will log `[Aragora] NEXT_PUBLIC_API_URL not set` if using localhost fallback

**Example Production Configuration:**
```bash
# OAuth (required in production)
GOOGLE_OAUTH_CLIENT_ID=1234567890-abc.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
GOOGLE_OAUTH_REDIRECT_URI=https://api.aragora.ai/api/auth/oauth/google/callback
OAUTH_SUCCESS_URL=https://aragora.ai/auth/success
OAUTH_ERROR_URL=https://aragora.ai/auth/error
OAUTH_ALLOWED_REDIRECT_HOSTS=aragora.ai,api.aragora.ai,www.aragora.ai

# Frontend URLs (required for deployed frontend)
NEXT_PUBLIC_API_URL=https://api.aragora.ai
NEXT_PUBLIC_WS_URL=wss://api.aragora.ai
```

### OAuth Runtime Controls

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OAUTH_STATE_TTL_SECONDS` | Optional | OAuth state TTL (seconds) | `600` |
| `OAUTH_MAX_STATES` | Optional | Max in-memory OAuth states | `10000` |

## AI Provider Keys

At least one AI provider key is required.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | One required | Anthropic Claude API key | - |
| `OPENAI_API_KEY` | One required | OpenAI API key | - |
| `GEMINI_API_KEY` | Optional | Google Gemini API key | - |
| `GOOGLE_API_KEY` | Optional | Alias for `GEMINI_API_KEY` | - |
| `XAI_API_KEY` | Optional | Grok/XAI API key | - |
| `GROK_API_KEY` | Optional | Alias for XAI_API_KEY | - |
| `MISTRAL_API_KEY` | Optional | Mistral AI API key (Large, Codestral) | - |
| `OPENROUTER_API_KEY` | Optional | OpenRouter for multi-model access | - |
| `DEEPSEEK_API_KEY` | Optional | DeepSeek CLI key (for `deepseek-cli`) | - |

**Note:** Never commit your `.env` file. It's gitignored for security.

### OpenRouter Models

OpenRouter provides access to multiple models through a single API:
- DeepSeek (V3, R1 Reasoner)
- Llama (Meta's open models)
- Mistral (also available via direct `MISTRAL_API_KEY`)
- Qwen (Alibaba's code and reasoning models)
- Yi (01.AI's balanced models)

See [OpenRouter docs](https://openrouter.ai/docs) for available models.

### Mistral Direct API

For best performance with Mistral models, use the direct API:
- `mistral-api` agent uses `MISTRAL_API_KEY` directly
- `codestral` agent for code-specialized tasks
- Falls back to OpenRouter if direct API fails

## Web Research (Experimental)

Enable external web research during debates (requires `aragora[research]`):

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `TAVILY_API_KEY` | Optional | Tavily search API key for web research | - |

## Ollama (Local Models)

Run AI models locally with Ollama.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OLLAMA_HOST` | Optional | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Optional | Default model name | `llama2` |

**Usage:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Set in .env (optional - defaults work for local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

## LM Studio (Local Models)

Run local LLMs through LM Studio's OpenAI-compatible server.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `LM_STUDIO_HOST` | Optional | LM Studio base URL | `http://localhost:1234` |

**Usage:**
```bash
# Start LM Studio server with a model loaded
# Default endpoint: http://localhost:1234/v1
LM_STUDIO_HOST=http://localhost:1234
```

## Persistence (Supabase)

Optional but recommended for production.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `SUPABASE_URL` | Optional | Supabase project URL | - |
| `SUPABASE_KEY` | Optional | Supabase service key | - |

Enables:
- Historical debate storage
- Cross-session learning
- Live dashboard at aragora.ai

## Database Connection (PostgreSQL/SQLite)

Use `DATABASE_URL` for managed Postgres, or set backend-specific settings for local control.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Optional | Postgres connection string (primary) | - |
| `ARAGORA_DATABASE_URL` | Optional | Legacy alias for `DATABASE_URL` | - |
| `ARAGORA_DB_BACKEND` | Optional | Backend: `sqlite`, `postgres`, `postgresql` | `sqlite` |
| `ARAGORA_DB_MODE` | Optional | Database layout: `legacy` or `consolidated` | `legacy` |
| `ARAGORA_DB_TIMEOUT` | Optional | Connection timeout (seconds) | `30` |
| `ARAGORA_DB_POOL_SIZE` | Optional | Connection pool size | `10` |
| `ARAGORA_DB_POOL_MAX_OVERFLOW` | Optional | Extra pool connections | `5` |
| `ARAGORA_DB_POOL_OVERFLOW` | Optional | Legacy alias for overflow (settings) | - |
| `ARAGORA_DB_POOL_TIMEOUT` | Optional | Pool wait timeout (seconds) | `30` |
| `ARAGORA_SQLITE_PATH` | Optional | SQLite path for the DB backend | `aragora.db` |
| `ARAGORA_SQLITE_POOL_SIZE` | Optional | SQLite pool size (storage backend) | `10` |
| `ARAGORA_PG_HOST` | Optional | Postgres host | `localhost` |
| `ARAGORA_PG_PORT` | Optional | Postgres port | `5432` |
| `ARAGORA_PG_DATABASE` | Optional | Postgres database name | `aragora` |
| `ARAGORA_PG_USER` | Optional | Postgres user | `aragora` |
| `ARAGORA_PG_PASSWORD` | Optional | Postgres password | - |
| `ARAGORA_PG_SSL_MODE` | Optional | Postgres SSL mode | `require` |
| `ARAGORA_POSTGRESQL_POOL_SIZE` | Optional | Postgres pool size (storage backend) | `5` |
| `ARAGORA_POSTGRESQL_POOL_MAX_OVERFLOW` | Optional | Postgres overflow (storage backend) | `10` |

Note: `ARAGORA_DB_MODE` defaults to `legacy` in the legacy config, while
`aragora.persistence.db_config` defaults to `consolidated` if unset. Set it
explicitly to avoid ambiguity. The storage backend also honors the
`ARAGORA_SQLITE_POOL_SIZE` / `ARAGORA_POSTGRESQL_*` pool settings; set them
explicitly if you need consistent pooling across subsystems.

## Server Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_API_URL` | Optional | API base URL for CLI/SDK clients | `http://localhost:8080` |
| `ARAGORA_ENV` | Recommended | `development` or `production` | `development` |
| `ARAGORA_ENVIRONMENT` | Optional | Alias used by billing/auth | `development` |
| `ARAGORA_API_TOKEN` | Optional | Enable token auth | Disabled |
| `ARAGORA_TOKEN_TTL` | Optional | Token lifetime (seconds) | `3600` |
| `ARAGORA_WS_MAX_MESSAGE_SIZE` | Optional | Max WebSocket message size | `65536` |
| `ARAGORA_WS_HEARTBEAT` | Optional | WebSocket heartbeat interval (seconds) | `30` |
| `ARAGORA_DEFAULT_HOST` | Optional | Fallback host for link generation | `localhost:8080` |

## Debate Defaults

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_DEFAULT_ROUNDS` | Optional | Default debate rounds | `3` |
| `ARAGORA_MAX_ROUNDS` | Optional | Max debate rounds | `10` |
| `ARAGORA_DEFAULT_CONSENSUS` | Optional | Consensus mode | `hybrid` |
| `ARAGORA_DEBATE_TIMEOUT` | Optional | Debate timeout (seconds) | `900` |
| `ARAGORA_AGENT_TIMEOUT` | Optional | Per-agent timeout (seconds) | `240` |

## Agent Defaults

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_DEFAULT_AGENTS` | Optional | Default agent list when none specified | `grok,anthropic-api,openai-api,deepseek,mistral-api,gemini,qwen-max,kimi` |
| `ARAGORA_STREAMING_AGENTS` | Optional | Agents allowed for streaming responses | `grok,anthropic-api,openai-api,mistral-api` |

## Streaming Controls

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_STREAM_BUFFER_SIZE` | Optional | Max SSE buffer size (bytes) | `10485760` |
| `ARAGORA_STREAM_CHUNK_TIMEOUT` | Optional | Timeout between stream chunks (seconds) | `30` |

## WebSocket & Audience Limits

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_TRUSTED_PROXIES` | Optional | Comma-separated proxy IPs for client IP resolution | `127.0.0.1,::1,localhost` |
| `ARAGORA_WS_CONN_RATE` | Optional | WS connections per IP per minute | `30` |
| `ARAGORA_WS_MAX_PER_IP` | Optional | Max concurrent WS connections per IP | `10` |
| `ARAGORA_WS_MSG_RATE` | Optional | WS messages per second per connection | `10` |
| `ARAGORA_WS_MSG_BURST` | Optional | WS message burst size | `20` |
| `ARAGORA_AUDIENCE_INBOX_MAX_SIZE` | Optional | Audience inbox queue size | `1000` |
| `ARAGORA_MAX_EVENT_QUEUE_SIZE` | Optional | Event queue size (server) | `10000` |

## Reserved / Not Yet Wired

These variables exist in the settings schema but are not currently wired into runtime behavior.

| Variable | Description | Default | Status |
|----------|-------------|---------|--------|
| `ARAGORA_MAX_CONTEXT_CHARS` | Max context length for truncation (chars) | `100000` | Planned |
| `ARAGORA_MAX_MESSAGE_CHARS` | Max message length for truncation (chars) | `50000` | Planned |
| `ARAGORA_LOCAL_FALLBACK_ENABLED` | Enable local LLM fallback in provider chains | `false` | Planned |
| `ARAGORA_LOCAL_FALLBACK_PRIORITY` | Prefer local LLMs over OpenRouter | `false` | Planned |

**Note:** `aragora serve` runs HTTP on port 8080 and WebSocket on port 8765 by default. The WebSocket server accepts `/` or `/ws`. For single-port deployments, embed `AiohttpUnifiedServer` (advanced).

## Legacy/Deployment Host & Port

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_HOST` | Optional | Legacy bind host used by deployment templates | `0.0.0.0` |
| `ARAGORA_PORT` | Optional | Legacy HTTP port used by deployment templates | `8080` |

These are not read by the CLI server directly; prefer `aragora serve --api-port/--ws-port` in local dev.

### Environment Mode

Set `ARAGORA_ENVIRONMENT=production` for production deployments. This enables:
- Strict JWT secret validation (required, minimum 32 characters)
- Disabled unsafe JWT fallbacks
- Blocked format-only API key validation
- Stricter security defaults

## Data Directory

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_DATA_DIR` | Recommended | Directory for all runtime data (databases, etc.) | `.nomic` |

All databases are stored under this directory:
- `agent_elo.db` - ELO rankings
- `continuum.db` - Memory storage
- `consensus_memory.db` - Consensus records
- `token_blacklist.db` - Revoked JWT tokens
- And others...

Related directories:
- `ARAGORA_NOMIC_DIR` - Legacy alias used by some migration tooling (defaults to `.nomic`)
- `ARAGORA_STORAGE_DIR` - Non-DB runtime artifacts (plugins, reviews, modes) default to `.aragora`

**Production recommended:** `/var/lib/aragora` or `~/.aragora` for `ARAGORA_DATA_DIR`

Use `aragora.config.resolve_db_path()` to keep legacy SQLite files under
`ARAGORA_DATA_DIR`. For consolidated mapping, use
`aragora.persistence.db_config.get_db_path()`.

### Cleanup (repo root artifacts)

If you ran Aragora in the repo root, stray `.db` files may land there. Move them under `ARAGORA_DATA_DIR` with:

```bash
scripts/cleanup_runtime_artifacts.sh
```

## CORS Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_ALLOWED_ORIGINS` | Optional | Comma-separated allowed origins | See below |

Default origins:
```
http://localhost:3000,http://localhost:8080,
http://127.0.0.1:3000,http://127.0.0.1:8080,
https://aragora.ai,https://www.aragora.ai,
https://api.aragora.ai
```

Example:
```bash
ARAGORA_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com
```

## Webhook Integration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_WEBHOOKS` | Optional | JSON array of webhook configs | - |
| `ARAGORA_WEBHOOKS_CONFIG` | Optional | Path to JSON config file | - |
| `ARAGORA_WEBHOOK_QUEUE_SIZE` | Optional | Max queued events | `1000` |
| `ARAGORA_WEBHOOK_ALLOW_LOCALHOST` | Optional | Allow localhost webhook targets (dev only) | `false` |
| `ARAGORA_WEBHOOK_WORKERS` | Optional | Max concurrent deliveries | `10` |
| `ARAGORA_WEBHOOK_MAX_RETRIES` | Optional | Delivery retry attempts | `3` |
| `ARAGORA_WEBHOOK_RETRY_DELAY` | Optional | Initial retry delay (seconds) | `1.0` |
| `ARAGORA_WEBHOOK_MAX_RETRY_DELAY` | Optional | Max retry delay (seconds) | `60.0` |
| `ARAGORA_WEBHOOK_TIMEOUT` | Optional | Request timeout (seconds) | `30.0` |

`ARAGORA_WEBHOOKS` and `ARAGORA_WEBHOOKS_CONFIG` accept a JSON array of configs with:
`name`, `url`, optional `secret`, optional `event_types`, and optional `loop_ids`.

## Slack Integration (Server)

Configure Slack slash commands and outbound notifications.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `SLACK_SIGNING_SECRET` | Optional | Verify Slack request signatures | - |
| `SLACK_BOT_TOKEN` | Optional | Bot token for Slack API calls | - |
| `SLACK_WEBHOOK_URL` | Optional | Outbound Slack webhook URL | - |

## Rate Limiting

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_RATE_LIMIT` | Optional | Requests per minute per token | `60` |
| `ARAGORA_IP_RATE_LIMIT` | Optional | Requests per minute per IP | `120` |
| `ARAGORA_BURST_MULTIPLIER` | Optional | Burst multiplier for short spikes | `2.0` |
| `ARAGORA_REDIS_URL` | Optional | Redis URL for distributed rate limits | `redis://localhost:6379/0` |
| `REDIS_URL` | Optional | Legacy Redis URL used by queues/oauth/token revocation | `redis://localhost:6379` |
| `ARAGORA_REDIS_KEY_PREFIX` | Optional | Redis key prefix | `aragora:ratelimit:` |
| `ARAGORA_REDIS_TTL` | Optional | Redis TTL for limiter keys (seconds) | `120` |
| `ARAGORA_REDIS_MAX_CONNECTIONS` | Optional | Redis connection pool max size | `50` |
| `ARAGORA_REDIS_SOCKET_TIMEOUT` | Optional | Redis socket timeout (seconds) | `5.0` |
| `ARAGORA_RATE_LIMIT_FAIL_OPEN` | Optional | Allow requests if Redis is down (`true`/`false`) | `false` |
| `ARAGORA_REDIS_FAILURE_THRESHOLD` | Optional | Failures before Redis limiter disables (count) | `3` |

## Request Timeout Middleware

Controls HTTP request timeouts to prevent hanging requests and cascading failures.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_REQUEST_TIMEOUT` | Optional | Default request timeout (seconds) | `30` |
| `ARAGORA_SLOW_REQUEST_TIMEOUT` | Optional | Timeout for slow endpoints like debates, broadcasts (seconds) | `60` |
| `ARAGORA_MAX_REQUEST_TIMEOUT` | Optional | Maximum allowed timeout (seconds) | `300` |
| `ARAGORA_TIMEOUT_WORKERS` | Optional | Thread pool size for sync timeout operations | `4` |

**Slow Endpoint Patterns** (automatically use `ARAGORA_SLOW_REQUEST_TIMEOUT`):
- `/api/debates/` - Debate orchestration
- `/api/broadcast/` - Audio/video generation
- `/api/evidence/` - Evidence collection
- `/api/gauntlet/` - Comprehensive testing

**Per-endpoint Overrides:**
```python
from aragora.server.middleware.timeout import configure_timeout

configure_timeout(
    endpoint_overrides={
        "/api/debates/run": 120.0,  # 2 minutes for debate runs
        "/api/broadcast/generate": 180.0,  # 3 minutes for video generation
    }
)
```

## Billing & Authentication

JWT authentication and Stripe integration for paid tiers.

### JWT Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_JWT_SECRET` | **Required (prod)** | Secret key for JWT signing (min 32 chars) | - |
| `ARAGORA_JWT_SECRET_PREVIOUS` | Optional | Previous secret for rotation | - |
| `ARAGORA_JWT_SECRET_ROTATED_AT` | Optional | Unix timestamp of rotation | - |
| `ARAGORA_JWT_ROTATION_GRACE_HOURS` | Optional | Grace period for previous secret | `24` |
| `ARAGORA_JWT_EXPIRY_HOURS` | Optional | Access token expiry (max 168h/7d) | `24` |
| `ARAGORA_REFRESH_TOKEN_EXPIRY_DAYS` | Optional | Refresh token expiry (max 90d) | `30` |
| `ARAGORA_ALLOW_FORMAT_ONLY_API_KEYS` | Optional | Allow API key format-only validation (dev only) | `0` |

**Security Notes:**
- In **production** (`ARAGORA_ENVIRONMENT=production`), `ARAGORA_JWT_SECRET` is **required** and must be at least 32 characters.
- Generate a secure secret: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- In other environments, set `ARAGORA_JWT_SECRET` if you use auth endpoints (missing secrets raise config errors).
- `ARAGORA_JWT_SECRET_PREVIOUS` is only honored if `ARAGORA_JWT_SECRET_ROTATED_AT` is set.
- Set `ARAGORA_JWT_ROTATION_GRACE_HOURS` to control the previous-secret window.
- `ARAGORA_ALLOW_FORMAT_ONLY_API_KEYS` is blocked in production regardless of setting.

### Token Blacklist Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_BLACKLIST_BACKEND` | Optional | Backend type: `memory`, `sqlite`, `redis` | `sqlite` |
| `ARAGORA_REDIS_URL` | For redis | Redis connection URL | `redis://localhost:6379/0` |

- **memory**: Fast but not persistent; use for development only
- **sqlite**: Default; persists to `ARAGORA_DATA_DIR/token_blacklist.db`
- **redis**: Use for multi-instance deployments (requires `redis` package)

### Stripe Integration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `STRIPE_SECRET_KEY` | For billing | Stripe API secret key | - |
| `STRIPE_WEBHOOK_SECRET` | For billing | Webhook signing secret | - |
| `STRIPE_PRICE_STARTER` | For billing | Price ID for Starter tier | - |
| `STRIPE_PRICE_PROFESSIONAL` | For billing | Price ID for Professional tier | - |
| `STRIPE_PRICE_ENTERPRISE` | For billing | Price ID for Enterprise tier | - |

See [BILLING.md](./BILLING.md) for subscription tiers and usage tracking.

### Billing Notifications

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SMTP_HOST` | Optional | SMTP server host | - |
| `ARAGORA_SMTP_PORT` | Optional | SMTP server port | `587` |
| `ARAGORA_SMTP_USER` | Optional | SMTP username | - |
| `ARAGORA_SMTP_PASSWORD` | Optional | SMTP password | - |
| `ARAGORA_SMTP_FROM` | Optional | From email address | `billing@aragora.ai` |
| `ARAGORA_NOTIFICATION_WEBHOOK` | Optional | Webhook URL for billing notifications | - |
| `ARAGORA_PAYMENT_GRACE_DAYS` | Optional | Days before downgrade after payment failure | `10` |
| `ARAGORA_ALLOW_INSECURE_PASSWORDS` | Optional | Allow weak passwords (dev only) | `0` |

## SSO/Enterprise Authentication

Single Sign-On configuration for enterprise authentication. Supports OIDC and SAML 2.0.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SSO_ENABLED` | No | Enable SSO authentication | `false` |
| `ARAGORA_SSO_PROVIDER_TYPE` | If SSO enabled | Provider type: `oidc`, `saml`, `azure_ad`, `okta`, `google` | `oidc` |
| `ARAGORA_SSO_CALLBACK_URL` | If SSO enabled | Callback URL for auth response (must be HTTPS in production) | - |
| `ARAGORA_SSO_ENTITY_ID` | If SSO enabled | Service provider entity ID | - |

### OIDC Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SSO_CLIENT_ID` | OIDC | OAuth client ID | - |
| `ARAGORA_SSO_CLIENT_SECRET` | OIDC | OAuth client secret | - |
| `ARAGORA_SSO_ISSUER_URL` | OIDC | OIDC issuer URL (e.g., `https://login.microsoftonline.com/tenant/v2.0`) | - |
| `ARAGORA_SSO_AUTH_ENDPOINT` | Optional | Override authorization endpoint | Auto-discovered |
| `ARAGORA_SSO_TOKEN_ENDPOINT` | Optional | Override token endpoint | Auto-discovered |
| `ARAGORA_SSO_USERINFO_ENDPOINT` | Optional | Override userinfo endpoint | Auto-discovered |
| `ARAGORA_SSO_JWKS_URI` | Optional | Override JWKS URI | Auto-discovered |
| `ARAGORA_SSO_SCOPES` | Optional | OAuth scopes | `openid,email,profile` |

### SAML Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SSO_IDP_ENTITY_ID` | SAML | IdP entity ID | - |
| `ARAGORA_SSO_IDP_SSO_URL` | SAML | IdP SSO URL | - |
| `ARAGORA_SSO_IDP_SLO_URL` | Optional | IdP logout URL | - |
| `ARAGORA_SSO_IDP_CERTIFICATE` | SAML | IdP X.509 certificate (PEM format) | - |
| `ARAGORA_SSO_SP_CERTIFICATE` | Optional | SP X.509 certificate for signed requests (PEM) | - |
| `ARAGORA_SSO_SP_PRIVATE_KEY` | Optional | SP private key for signed requests (PEM) | - |

### SSO Options

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SSO_ALLOWED_DOMAINS` | Optional | Comma-separated allowed email domains | - (all allowed) |
| `ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS` | Optional | Allowed redirect hosts for SSO callbacks | - |
| `ARAGORA_SSO_AUTO_PROVISION` | Optional | Auto-create users on first login | `true` |
| `ARAGORA_SSO_SESSION_DURATION` | Optional | Session duration in seconds (300-604800) | `28800` (8h) |

**Security Notes:**
- In **production** (`ARAGORA_ENV=production`), callback URLs must use HTTPS
- SAML in production requires `python3-saml` package for signature validation
- Certificates must be in PEM format (starting with `-----BEGIN`)

**Example OIDC Configuration (Azure AD):**
```bash
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=azure_ad
ARAGORA_SSO_CLIENT_ID=your-app-client-id
ARAGORA_SSO_CLIENT_SECRET=your-client-secret
ARAGORA_SSO_ISSUER_URL=https://login.microsoftonline.com/your-tenant-id/v2.0
ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback
ARAGORA_SSO_ENTITY_ID=https://your-app.example.com
ARAGORA_SSO_ALLOWED_DOMAINS=yourcompany.com
```

See [SSO_SETUP.md](./SSO_SETUP.md) for detailed provider-specific setup instructions.

## Embedding Providers

For semantic search and memory retrieval.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_EMBEDDING_MODEL` | Optional | OpenAI embedding model | `text-embedding-3-small` |

Currently uses OpenAI or Gemini embeddings based on available API keys.

## Broadcast / TTS

Configure audio generation backends for broadcasts.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_TTS_ORDER` | Optional | Comma-separated backend priority | `elevenlabs,xtts,edge-tts,pyttsx3` |
| `ARAGORA_TTS_BACKEND` | Optional | Force a specific backend | - |
| `ARAGORA_ELEVENLABS_API_KEY` | Optional | ElevenLabs API key | - |
| `ARAGORA_ELEVENLABS_MODEL_ID` | Optional | ElevenLabs model ID | `eleven_multilingual_v2` |
| `ARAGORA_ELEVENLABS_VOICE_ID` | Optional | Default ElevenLabs voice ID | - |
| `ARAGORA_ELEVENLABS_VOICE_MAP` | Optional | JSON map of speaker→voice ID | - |
| `ARAGORA_XTTS_MODEL_PATH` | Optional | Coqui XTTS model name/path | `tts_models/multilingual/multi-dataset/xtts_v2` |
| `ARAGORA_XTTS_DEVICE` | Optional | XTTS device (`auto`, `cuda`, `cpu`) | `auto` |
| `ARAGORA_XTTS_LANGUAGE` | Optional | XTTS language code | `en` |
| `ARAGORA_XTTS_SPEAKER_WAV` | Optional | Default XTTS speaker WAV path | - |
| `ARAGORA_XTTS_SPEAKER_WAV_MAP` | Optional | JSON map of speaker→WAV path | - |

Notes:
- `ELEVENLABS_API_KEY` is also accepted as an alias for `ARAGORA_ELEVENLABS_API_KEY`.
- Use `ARAGORA_TTS_ORDER` to prioritize ElevenLabs or XTTS ahead of edge-tts.

## Social Media APIs (Pulse Module)

For trending topics and real-time context in debates. These power the Pulse ingestors.

| Variable | Required | Description | Source |
|----------|----------|-------------|--------|
| `TWITTER_BEARER_TOKEN` | Optional | Twitter/X API v2 Bearer token for trending topics | [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard) |
| `ARAGORA_ALLOWED_OAUTH_HOSTS` | Optional | Comma-separated allowed hosts for social OAuth redirects | `localhost:8080,127.0.0.1:8080` (dev) |

**No credentials needed:**
- **Reddit** - Uses public JSON API (`reddit.com/.json`)
- **Hacker News** - Uses public Firebase API (`hacker-news.firebaseio.com`)

These services are automatically enabled when the pulse module loads.

### Getting Twitter API Access

1. Create a developer account at [developer.twitter.com](https://developer.twitter.com)
2. Create a new project and app
3. Generate a Bearer Token (read-only access is sufficient)
4. Add to your `.env`:
   ```bash
   TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAx...
   ```

### Pulse Module Usage

The pulse module fetches trending topics that can inform debate context:

```python
from aragora.pulse import PulseManager

pulse = PulseManager()
trends = await pulse.get_trending()  # Returns combined trends from all sources
```

## Formal Verification

> **Note:** These variables are defined but not yet actively used in the codebase.

| Variable | Required | Description | Default | Status |
|----------|----------|-------------|---------|--------|
| `Z3_TIMEOUT` | Optional | Z3 solver timeout (seconds) | `30` | Planned |
| `LEAN_PATH` | Optional | Path to Lean 4 installation | Auto-detect | Planned |

## Telemetry Configuration

Controls observation levels for debug and production modes.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_TELEMETRY_LEVEL` | Optional | Telemetry level (SILENT/DIAGNOSTIC/CONTROLLED/SPECTACLE) | `CONTROLLED` |

Levels:
- `SILENT` (0): No telemetry broadcast
- `DIAGNOSTIC` (1): Internal diagnostics only
- `CONTROLLED` (2): Redacted telemetry (default, secrets filtered)
- `SPECTACLE` (3): Full transparency (development only)

## Belief Network Settings

For belief propagation analysis during debates.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_BELIEF_MAX_ITERATIONS` | Optional | Max iterations for belief convergence | `100` |
| `ARAGORA_BELIEF_CONVERGENCE_THRESHOLD` | Optional | Convergence epsilon | `0.001` |

## Queue Settings

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_USER_EVENT_QUEUE_SIZE` | Optional | User event queue buffer size | `100` |

## Broadcast (Audio/Podcast)

Configuration for debate-to-podcast conversion.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_AUDIO_DIR` | Optional | Audio storage directory | `.nomic/audio/` |
| `ARAGORA_TTS_TIMEOUT` | Optional | TTS generation timeout (seconds) | `60` |
| `ARAGORA_TTS_RETRIES` | Optional | TTS retry attempts | `3` |

See [BROADCAST.md](./BROADCAST.md) for the complete audio pipeline documentation.

## Debug & Logging

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_DEBUG` | Optional | Enable debug logging | `false` |
| `ARAGORA_LOG_LEVEL` | Optional | Log level (DEBUG/INFO/WARN/ERROR) | `INFO` |

## Validation Rules

### API Keys
- Must be non-empty strings
- Validated on first API call
- Keys are not logged (security)

### Ports
- Must be integers 1-65535
- HTTP API: 8080 (default)
- WebSocket: 8765 (default, `/` or `/ws`)
- Single-port option: use `AiohttpUnifiedServer` (advanced)

### URLs
- Must be valid HTTPS URLs (for production)
- HTTP allowed for localhost development

## Example .env File

```bash
# Required: At least one AI provider
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx

# Optional: Additional providers
GEMINI_API_KEY=AIzaSy...
XAI_API_KEY=xai-xxx
MISTRAL_API_KEY=xxx
OPENROUTER_API_KEY=sk-or-xxx
DEEPSEEK_API_KEY=sk-xxx

# Optional: Local models
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Optional: Persistence
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Optional: Client defaults / auth
ARAGORA_API_URL=http://localhost:8080
ARAGORA_API_TOKEN=my-secret-token

# Optional: JWT Authentication
ARAGORA_JWT_SECRET=your-secure-secret-key
ARAGORA_JWT_EXPIRY_HOURS=24

# Optional: Stripe Billing
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_PRICE_STARTER=price_xxx
STRIPE_PRICE_PROFESSIONAL=price_xxx
STRIPE_PRICE_ENTERPRISE=price_xxx

# Optional: Redis (rate limiting, queues, oauth state)
ARAGORA_REDIS_URL=redis://localhost:6379/0
REDIS_URL=redis://localhost:6379

# Optional: Webhooks
ARAGORA_WEBHOOKS_CONFIG=/etc/aragora/webhooks.json

# Optional: Social Media (Pulse module)
TWITTER_BEARER_TOKEN=AAAA...  # For trending topics
```

## Troubleshooting

### "No API key found"
- Set at least one of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- Verify `.env` file is in project root

### "CORS blocked"
- Add your domain to `ARAGORA_ALLOWED_ORIGINS`
- Check for typos in origin URLs

### "WebSocket connection failed"
- Verify `--ws-port` (server) matches `NEXT_PUBLIC_WS_URL` (frontend) or your client URL
- Check firewall/proxy settings

### "Rate limit exceeded"
- Increase `ARAGORA_RATE_LIMIT` / `ARAGORA_IP_RATE_LIMIT`
- Or wait for rate limit window to reset

---

## SSL/TLS Configuration

Enable HTTPS for production deployments.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SSL_ENABLED` | Optional | Enable SSL/TLS | `false` |
| `ARAGORA_SSL_CERT` | If SSL enabled | Path to SSL certificate file | - |
| `ARAGORA_SSL_KEY` | If SSL enabled | Path to SSL private key file | - |

Example:
```bash
ARAGORA_SSL_ENABLED=true
ARAGORA_SSL_CERT=/etc/ssl/certs/aragora.pem
ARAGORA_SSL_KEY=/etc/ssl/private/aragora-key.pem
```

### Self-signed certificate for development
```bash
# Generate a self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Use with Aragora
ARAGORA_SSL_ENABLED=true
ARAGORA_SSL_CERT=cert.pem
ARAGORA_SSL_KEY=key.pem
```

---

## Deployment Tuning Guides

### High-Load Deployments

For production systems handling many concurrent debates:

```bash
# Rate limiting - increase for high-traffic APIs
ARAGORA_RATE_LIMIT=200          # 200 req/min per token
ARAGORA_IP_RATE_LIMIT=500       # 500 req/min per IP

# Debate limits
ARAGORA_MAX_AGENTS_PER_DEBATE=8 # Limit agents per debate
ARAGORA_MAX_CONCURRENT_DEBATES=50  # Allow more parallel debates

# WebSocket settings
ARAGORA_WS_MAX_MESSAGE_SIZE=131072  # 128KB for large messages
ARAGORA_WS_HEARTBEAT=15            # More frequent heartbeats

# Database timeouts
ARAGORA_DB_TIMEOUT=60.0            # Longer timeout for complex queries

# Cache TTLs - shorter for freshness
ARAGORA_CACHE_LEADERBOARD=60       # 1 minute leaderboard cache
ARAGORA_CACHE_AGENT_PROFILE=120    # 2 minute profile cache
```

### Development Mode

For local development with faster iteration:

```bash
# Debug output
ARAGORA_DEBUG=true
ARAGORA_LOG_LEVEL=DEBUG

# Disable SSL for localhost
ARAGORA_SSL_ENABLED=false

# Lower timeouts for faster feedback
ARAGORA_DEBATE_TIMEOUT=120         # 2 minute debate timeout
ARAGORA_DB_TIMEOUT=10.0            # Quick database timeout

# Generous rate limits
ARAGORA_RATE_LIMIT=1000
ARAGORA_IP_RATE_LIMIT=1000

# Full telemetry for debugging
ARAGORA_TELEMETRY_LEVEL=SPECTACLE
```

### Testing Configuration

For running test suites:

```bash
# Use in-memory or test databases
ARAGORA_DB_ELO=:memory:
ARAGORA_DB_MEMORY=:memory:

# Short timeouts for fast tests
ARAGORA_DB_TIMEOUT=5.0
ARAGORA_DEBATE_TIMEOUT=30

# Disable external services
# (Don't set API keys to skip external API tests)

# Disable SSL
ARAGORA_SSL_ENABLED=false

# Silent telemetry
ARAGORA_TELEMETRY_LEVEL=SILENT
```

---

## Configuration Validation

Aragora validates configuration at startup. Check configuration with:

```python
from aragora.config import validate_configuration

# Non-strict: logs warnings, returns validation result
result = validate_configuration(strict=False)
print(result["valid"])       # True if no errors
print(result["warnings"])    # List of warnings
print(result["config_summary"])  # Current config values

# Strict: raises ConfigurationError on errors
from aragora.config import validate_configuration, ConfigurationError
try:
    validate_configuration(strict=True)
except ConfigurationError as e:
    print(f"Config error: {e}")
```

### Validation Checks

- **Rate limits**: Must be positive integers
- **Timeouts**: Must be positive numbers
- **SSL paths**: Must exist if SSL enabled
- **API keys**: Warning if none configured (error in strict mode)
