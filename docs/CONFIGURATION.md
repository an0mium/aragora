# Aragora Configuration Reference

Complete reference for all configuration options. For quick setup, see `.env.example`.

## Quick Start

```bash
# Copy the example config
cp .env.example .env

# Minimum required: at least one AI provider key
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Start the server
python -m aragora.server.unified_server
```

## Production Checklist

Before deploying to production, ensure:

- [ ] `ARAGORA_ENVIRONMENT=production`
- [ ] `JWT_SECRET` set (min 32 chars): `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- [ ] `ARAGORA_ALLOWED_ORIGINS` set to your domain(s)
- [ ] At least one AI provider API key configured
- [ ] `DATABASE_URL` or `SUPABASE_URL` for persistence
- [ ] `REDIS_URL` for multi-instance deployments
- [ ] `ARAGORA_CSP_REPORT_ONLY=false` to enforce CSP
- [ ] Review rate limiting settings

Verify configuration:
```bash
python -m aragora.cli doctor
```

---

## 1. AI Providers

### Primary LLM APIs

At least one required for debates to function.

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | Anthropic Claude API key |
| `OPENAI_API_KEY` | Yes* | OpenAI GPT API key |
| `GEMINI_API_KEY` | Yes* | Google Gemini API key |
| `XAI_API_KEY` | Yes* | xAI Grok API key |
| `MISTRAL_API_KEY` | Yes* | Mistral AI API key |
| `OPENROUTER_API_KEY` | Yes* | OpenRouter multi-model API (fallback) |
| `DEEPSEEK_API_KEY` | No | DeepSeek direct API |

*At least one required

### Local Models

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LM_STUDIO_HOST` | `http://localhost:1234` | LM Studio server URL |

### Embeddings & Vector Search

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions |
| `VECTOR_BACKEND` | `memory` | Vector store backend |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate cluster URL |
| `WEAVIATE_API_KEY` | - | Weaviate Cloud API key |
| `ARAGORA_WEAVIATE_ENABLED` | `false` | Enable Weaviate |

---

## 2. Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_PORT` | `8080` | HTTP/WebSocket server port |
| `ARAGORA_BIND_HOST` | `127.0.0.1` | Server bind address |
| `ARAGORA_ENVIRONMENT` | `development` | Environment mode |
| `ARAGORA_DATA_DIR` | `.nomic` | Data directory for databases |
| `ARAGORA_INSTANCE_COUNT` | `1` | Number of server instances |

---

## 3. Database & Persistence

### Database Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_DB_BACKEND` | `sqlite` | Backend: `sqlite` or `postgres` |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `ARAGORA_POSTGRES_DSN` | - | Legacy alias for `DATABASE_URL` |
| `ARAGORA_DATABASE_URL` | - | Legacy alias for `DATABASE_URL` |
| `ARAGORA_SQLITE_PATH` | `aragora.db` | SQLite file path |
| `ARAGORA_POLICY_STORE_BACKEND` | `auto` | Policy store backend: `sqlite` or `postgres` (defaults to `ARAGORA_DB_BACKEND`) |
| `ARAGORA_AUDIT_STORE_BACKEND` | `auto` | Audit log backend: `sqlite` or `postgres` (defaults to `ARAGORA_DB_BACKEND`) |
| `ARAGORA_REQUIRE_DISTRIBUTED` | `auto` | Enforce distributed stores in production (fail closed on SQLite/file) |
| `ARAGORA_STORAGE_MODE` | `auto` | Force storage mode: `postgres`, `redis`, `sqlite`, `file` |

**PostgreSQL Connection String Examples:**
```bash
# Supabase
DATABASE_URL=postgresql://postgres.[ref]:[pass]@aws-0-us-west-1.pooler.supabase.com:6543/postgres

# AWS RDS
DATABASE_URL=postgresql://admin:password@mydb.xxxx.us-west-2.rds.amazonaws.com:5432/aragora

# Local
DATABASE_URL=postgresql://user:pass@localhost:5432/aragora
```

**Distributed Store Enforcement:**
- In production, `ARAGORA_REQUIRE_DISTRIBUTED=true` will fail closed if a store falls back to SQLite or local files.
- Set `ARAGORA_STORAGE_MODE` to pin a backend across stores when validating production readiness.

### Connection Pooling (PostgreSQL)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_DB_POOL_SIZE` | `20` | Connection pool size |
| `ARAGORA_DB_POOL_MAX_OVERFLOW` | `15` | Max overflow connections |
| `ARAGORA_DB_POOL_TIMEOUT` | `30.0` | Acquisition timeout (seconds) |

### Supabase

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | - | Supabase project URL |
| `SUPABASE_KEY` | - | Supabase API key |
| `SUPABASE_SYNC_ENABLED` | `false` | Enable background sync |

---

## 4. Redis (Caching & Sessions)

Required for multi-instance deployments and distributed rate limiting.

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | - | Redis connection URL |
| `ARAGORA_REDIS_PASSWORD` | - | Redis password |
| `ARAGORA_REDIS_MAX_CONNECTIONS` | `50` | Max connections |
| `ARAGORA_REDIS_SOCKET_TIMEOUT` | `5.0` | Socket timeout |

### Redis Cluster

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_REDIS_CLUSTER_MODE` | `auto` | Cluster mode detection |
| `ARAGORA_REDIS_CLUSTER_NODES` | - | Cluster node addresses |
| `ARAGORA_REDIS_CLUSTER_READ_FROM_REPLICAS` | `true` | Read from replicas |

### Distributed State (Horizontal Scaling)

For multi-instance deployments where debate state needs to be shared across servers.

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_STATE_BACKEND` | `memory` | Backend: `memory` (single instance) or `redis` (multi-instance) |
| `ARAGORA_REDIS_URL` | `redis://localhost:6379` | Redis URL for state storage |
| `ARAGORA_INSTANCE_ID` | `instance-{pid}` | Unique instance identifier for event deduplication |

**When to use Redis state:**
- Running multiple server instances behind a load balancer
- Need debate continuation after server restart
- Cross-instance WebSocket event broadcasting

**Example multi-instance setup:**
```bash
# Instance 1
ARAGORA_STATE_BACKEND=redis
ARAGORA_REDIS_URL=redis://redis-server:6379
ARAGORA_INSTANCE_ID=server-1
ARAGORA_PORT=8081

# Instance 2
ARAGORA_STATE_BACKEND=redis
ARAGORA_REDIS_URL=redis://redis-server:6379
ARAGORA_INSTANCE_ID=server-2
ARAGORA_PORT=8082
```

---

## 5. Authentication & Security

### JWT Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | - | **Required in production** (min 32 chars) |
| `ARAGORA_JWT_EXPIRY_HOURS` | `24` | Token expiry |
| `ARAGORA_REFRESH_TOKEN_EXPIRY_DAYS` | `30` | Refresh token expiry |

### API Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_API_TOKEN` | - | Simple token auth (dev) |
| `ARAGORA_AUTH_REQUIRED` | `false` | Require auth for WebSocket |

### Content Security Policy

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_ENABLE_CSP` | `true` | Enable CSP header |
| `ARAGORA_CSP_MODE` | `standard` | Mode: `api`, `standard`, `development` |
| `ARAGORA_CSP_REPORT_ONLY` | `true` | Report-only mode |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_ALLOWED_ORIGINS` | - | Comma-separated allowed origins |
| `ARAGORA_TRUSTED_PROXIES` | `127.0.0.1,::1,localhost` | Trusted proxy IPs |

---

## 6. Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_RATE_LIMIT` | `60` | Requests per minute |
| `ARAGORA_IP_RATE_LIMIT` | `120` | Per-IP rate limit |
| `ARAGORA_BURST_MULTIPLIER` | `2.0` | Burst multiplier |
| `ARAGORA_RATE_LIMIT_FAIL_OPEN` | `false` | Allow if Redis down |

### WebSocket Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_WS_CONN_RATE` | `30` | Connections per minute |
| `ARAGORA_WS_MAX_PER_IP` | `10` | Max connections per IP |
| `ARAGORA_WS_MSG_RATE` | `10` | Messages per second |

---

## 7. Timeouts

### Request Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_REQUEST_TIMEOUT` | `30` | Default request timeout |
| `ARAGORA_MAX_REQUEST_TIMEOUT` | `600` | Maximum allowed timeout |
| `ARAGORA_SLOW_REQUEST_TIMEOUT` | `120` | Slow request threshold |

### Debate Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_SLOW_DEBATE_THRESHOLD` | `30` | Slow debate detection (seconds) |
| `ARAGORA_CONTEXT_TIMEOUT` | `150.0` | Context gathering timeout |
| `ARAGORA_EVIDENCE_TIMEOUT` | `30.0` | Evidence collection timeout |

---

## 8. Observability

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_LOG_LEVEL` | `INFO` | Log level |
| `ARAGORA_LOG_FORMAT` | `json` | Format: `json` or `text` |
| `ARAGORA_LOG_FILE` | - | Log file path (stdout if not set) |
| `ARAGORA_LOG_MAX_BYTES` | `10485760` | Max log file size |
| `ARAGORA_LOG_BACKUP_COUNT` | `5` | Number of backups |

### Prometheus Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `METRICS_ENABLED` | `true` | Enable metrics endpoint |
| `METRICS_PORT` | `9090` | Metrics server port |
| `ARAGORA_METRICS_TOKEN` | - | Token to protect `/metrics` |

### OpenTelemetry

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Enable tracing |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector |
| `OTEL_SERVICE_NAME` | `aragora` | Service name |
| `OTEL_SAMPLE_RATE` | `1.0` | Sampling rate (0.0-1.0) |

### Sentry

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTRY_DSN` | - | Sentry DSN URL |
| `SENTRY_ENVIRONMENT` | `development` | Environment tag |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Trace sampling |

---

## 9. Billing (Stripe)

| Variable | Default | Description |
|----------|---------|-------------|
| `STRIPE_SECRET_KEY` | - | Stripe secret API key |
| `STRIPE_WEBHOOK_SECRET` | - | Webhook signing secret |
| `STRIPE_PRICE_STARTER` | - | Starter plan price ID |
| `STRIPE_PRICE_PROFESSIONAL` | - | Professional plan price ID |
| `STRIPE_PRICE_ENTERPRISE` | - | Enterprise plan price ID |

---

## 10. Chat Integrations

### Slack

| Variable | Description |
|----------|-------------|
| `SLACK_BOT_TOKEN` | Bot token (xoxb-...) |
| `SLACK_APP_TOKEN` | App token (xapp-...) |
| `SLACK_SIGNING_SECRET` | Request signing secret |

### Discord

| Variable | Description |
|----------|-------------|
| `DISCORD_BOT_TOKEN` | Bot token |
| `DISCORD_APPLICATION_ID` | Application ID |
| `DISCORD_PUBLIC_KEY` | Public key for verification |

### Microsoft Teams

| Variable | Description |
|----------|-------------|
| `TEAMS_APP_ID` | Application ID |
| `TEAMS_APP_PASSWORD` | Application password |
| `TEAMS_TENANT_ID` | Tenant ID |

---

## 11. Enterprise Integrations

### Snowflake

| Variable | Description |
|----------|-------------|
| `SNOWFLAKE_USER` | Username |
| `SNOWFLAKE_PASSWORD` | Password |
| `SNOWFLAKE_PRIVATE_KEY_PATH` | Key-pair auth path |
| `SNOWFLAKE_AUTHENTICATOR` | Auth method |

### Gmail

| Variable | Description |
|----------|-------------|
| `GMAIL_CLIENT_ID` | OAuth client ID |
| `GMAIL_CLIENT_SECRET` | OAuth client secret |

---

## 12. Text-to-Speech

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_TTS_BACKEND` | - | TTS backend selection |
| `ARAGORA_TTS_ORDER` | - | Backend priority order |
| `ARAGORA_ELEVENLABS_API_KEY` | - | ElevenLabs API key |
| `ARAGORA_ELEVENLABS_VOICE_ID` | - | Default voice ID |

---

## 13. Webhooks

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_NOTIFICATION_WEBHOOK` | - | Notification webhook URL |
| `ARAGORA_WEBHOOK_TIMEOUT` | `30.0` | Delivery timeout |
| `ARAGORA_WEBHOOK_MAX_RETRIES` | `3` | Max retries |
| `ARAGORA_WEBHOOK_ALLOW_LOCALHOST` | `false` | Allow localhost targets |

---

## Environment-Specific Configurations

### Development
```bash
ARAGORA_ENVIRONMENT=development
ARAGORA_LOG_FORMAT=text
ARAGORA_LOG_LEVEL=DEBUG
ARAGORA_CSP_REPORT_ONLY=true
```

### Production
```bash
ARAGORA_ENVIRONMENT=production
ARAGORA_LOG_FORMAT=json
ARAGORA_LOG_LEVEL=INFO
ARAGORA_CSP_REPORT_ONLY=false
JWT_SECRET=your-32-char-secret-here
ARAGORA_ALLOWED_ORIGINS=https://yourdomain.com
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Testing
```bash
ARAGORA_ENVIRONMENT=test
TESTING=1
ARAGORA_DB_BACKEND=sqlite
ARAGORA_DATA_DIR=/tmp/aragora-test
```

---

## Validation

Run the configuration doctor to validate your setup:

```bash
python -m aragora.cli doctor
```

This checks:
- Required environment variables
- Database connectivity
- AI provider availability
- Redis connectivity (if configured)
- SSL certificate validity (if enabled)

---

## See Also

- `.env.example` - Quick-start configuration template
- `.env.starter` - Minimal configuration for getting started
- `docs/ENVIRONMENT.md` - Legacy environment documentation
- `aragora/config/settings.py` - Pydantic settings schema
