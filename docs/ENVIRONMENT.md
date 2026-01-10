# Environment Variable Reference

Complete reference for all environment variables used by Aragora.

## Quick Start

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

## AI Provider Keys

At least one AI provider key is required.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | One required | Anthropic Claude API key | - |
| `OPENAI_API_KEY` | One required | OpenAI API key | - |
| `GEMINI_API_KEY` | Optional | Google Gemini API key | - |
| `XAI_API_KEY` | Optional | Grok/XAI API key | - |
| `GROK_API_KEY` | Optional | Alias for XAI_API_KEY | - |
| `OPENROUTER_API_KEY` | Optional | OpenRouter for multi-model access | - |
| `DEEPSEEK_API_KEY` | Optional | DeepSeek API key | - |

**Note:** Never commit your `.env` file. It's gitignored for security.

### OpenRouter Models

OpenRouter provides access to multiple models through a single API:
- DeepSeek (V3, Reasoner)
- Llama (Meta's open models)
- Mistral

See [OpenRouter docs](https://openrouter.ai/docs) for available models.

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

## Persistence (Supabase)

Optional but recommended for production.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `SUPABASE_URL` | Optional | Supabase project URL | - |
| `SUPABASE_KEY` | Optional | Supabase service key | - |

Enables:
- Historical debate storage
- Cross-session learning
- Live dashboard at live.aragora.ai

## Server Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `API_PORT` | Optional | HTTP server port | `8080` |
| `API_HOST` | Optional | Bind address | `0.0.0.0` |
| `ARAGORA_API_TOKEN` | Optional | Enable token auth | Disabled |
| `ARAGORA_TOKEN_TTL` | Optional | Token lifetime (seconds) | `3600` |
| `ARAGORA_WS_MAX_SIZE` | Optional | Max WebSocket message size | `65536` |

## CORS Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_ALLOWED_ORIGINS` | Optional | Comma-separated allowed origins | See below |

Default origins:
```
http://localhost:3000,http://localhost:8080,
http://127.0.0.1:3000,http://127.0.0.1:8080,
https://aragora.ai,https://www.aragora.ai,
https://live.aragora.ai,https://api.aragora.ai
```

Example:
```bash
ARAGORA_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com
```

## Webhook Integration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `WEBHOOK_URL` | Optional | External webhook endpoint | - |
| `WEBHOOK_SECRET` | Optional | HMAC secret for signing | - |
| `ARAGORA_WEBHOOK_QUEUE_SIZE` | Optional | Max queued events | `1000` |

## Rate Limiting

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `RATE_LIMIT_PER_MINUTE` | Optional | Requests per minute per token | `60` |
| `IP_RATE_LIMIT_PER_MINUTE` | Optional | Requests per minute per IP | `120` |

## Billing & Authentication

JWT authentication and Stripe integration for paid tiers.

### JWT Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_JWT_SECRET` | Recommended | Secret key for JWT signing | Auto-generated |
| `ARAGORA_JWT_EXPIRY_HOURS` | Optional | Access token expiry | `24` |
| `ARAGORA_REFRESH_TOKEN_EXPIRY_DAYS` | Optional | Refresh token expiry | `30` |
| `ARAGORA_ALLOW_FORMAT_ONLY_API_KEYS` | Optional | Allow API key format-only validation (dev/test) | `0` |

**Important:** Set `ARAGORA_JWT_SECRET` in production. Auto-generated secrets are invalidated on restart.

### Stripe Integration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `STRIPE_SECRET_KEY` | For billing | Stripe API secret key | - |
| `STRIPE_WEBHOOK_SECRET` | For billing | Webhook signing secret | - |
| `STRIPE_PRICE_STARTER` | For billing | Price ID for Starter tier | - |
| `STRIPE_PRICE_PROFESSIONAL` | For billing | Price ID for Professional tier | - |
| `STRIPE_PRICE_ENTERPRISE` | For billing | Price ID for Enterprise tier | - |

See [BILLING.md](./BILLING.md) for subscription tiers and usage tracking.

## Embedding Providers

For semantic search and memory retrieval.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_EMBEDDING_MODEL` | Optional | OpenAI embedding model | `text-embedding-3-small` |

Currently uses OpenAI or Gemini embeddings based on available API keys.

## Social Media APIs (Pulse Module)

For trending topics and real-time context in debates. These power the Pulse ingestors.

| Variable | Required | Description | Source |
|----------|----------|-------------|--------|
| `TWITTER_BEARER_TOKEN` | Optional | Twitter/X API v2 Bearer token for trending topics | [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard) |

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
- Unified server: 8080 (HTTP + WebSocket on /ws/debates)
- Legacy WebSocket: 8765 (deprecated, use unified server)

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
OPENROUTER_API_KEY=sk-or-xxx
DEEPSEEK_API_KEY=sk-xxx

# Optional: Local models
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Optional: Persistence
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Optional: Server config
API_PORT=8080
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

# Optional: Webhooks
WEBHOOK_URL=https://myserver.com/aragora-events
WEBHOOK_SECRET=hmac-secret

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
- Verify `API_PORT` matches your client config
- Check firewall/proxy settings

### "Rate limit exceeded"
- Increase `RATE_LIMIT_PER_MINUTE`
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
