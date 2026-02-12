# Authentication Guide

How to authenticate with the Aragora platform.

## Overview

Aragora supports three authentication methods:

| Method | Use case | Header format |
|--------|----------|---------------|
| **API Key** | SDK clients, CI/CD, automation | `Authorization: Bearer ara_...` |
| **JWT Token** | Web app sessions, SSO | `Authorization: Bearer eyJ...` |
| **HMAC Token** | Legacy integrations | `Authorization: Bearer <hmac-token>` |

Most developers should use **API keys**.

## API Keys

### Getting your first key

**Option A: CLI (recommended)**

```bash
# Login first
aragora auth login

# Create a named key
aragora auth create-key --name "my-dev-key"
# Output: ara_abc123...
```

**Option B: Web UI**

1. Navigate to Settings > API Tokens
2. Click "Create Token"
3. Copy the key (shown only once)

**Option C: Self-hosted (no auth)**

When running in offline/demo mode, authentication is optional:

```bash
python -m aragora.server.unified_server --offline
```

### Using your key

**Environment variable (recommended):**

```bash
export ARAGORA_API_KEY="ara_your_key_here"
```

```python
from aragora_sdk import AragoraClient

# from_env() reads ARAGORA_API_KEY automatically
client = AragoraClient.from_env()
```

**Explicit parameter:**

```python
client = AragoraClient(
    base_url="https://api.aragora.ai",
    api_key="ara_your_key_here",
)
```

**In .env file:**

```bash
# .env
ARAGORA_API_URL=http://localhost:8080
ARAGORA_API_KEY=ara_your_key_here
```

```python
from dotenv import load_dotenv
from aragora_sdk import AragoraClient

load_dotenv()
client = AragoraClient.from_env()
```

### Key scoping and permissions

API keys inherit the permissions of the user who created them. To restrict a key's access, create it with a specific role:

```bash
aragora auth create-key --name "read-only" --role viewer
```

Available roles: `admin`, `operator`, `analyst`, `viewer`, `api_consumer`, `auditor`, `guest`

### Key rotation

```bash
# List active keys
aragora auth list-keys

# Revoke a key
aragora auth revoke-key --name "old-key"

# Create a replacement
aragora auth create-key --name "new-key"
```

## LLM Provider Keys

Aragora uses LLM providers (Anthropic, OpenAI, etc.) for agent reasoning. These are separate from your Aragora API key.

```bash
# Set at least one provider
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude
export OPENAI_API_KEY="sk-..."          # GPT-4
export GEMINI_API_KEY="..."             # Gemini
export XAI_API_KEY="..."               # Grok

# Optional: fallback provider (used automatically on 429 rate limits)
export OPENROUTER_API_KEY="sk-or-..."
```

The server uses these keys server-side. SDK clients don't need LLM keys -- they authenticate to the Aragora API, which handles LLM calls internally.

## SSO / OAuth

For enterprise deployments, Aragora supports OIDC and SAML SSO:

```bash
# Configure OIDC provider
export ARAGORA_OIDC_ISSUER="https://your-idp.com"
export ARAGORA_OIDC_CLIENT_ID="aragora-app"
export ARAGORA_OIDC_CLIENT_SECRET="..."
```

Users authenticate via the web UI login page, which redirects to your identity provider. The server issues a JWT after successful SSO login.

## Multi-tenant authentication

In multi-tenant deployments, each tenant has isolated data and separate API keys. Tenant context is determined by the API key used:

```python
# Tenant A's key accesses Tenant A's data only
client_a = AragoraClient(base_url="...", api_key="ara_tenant_a_key")

# Tenant B's key accesses Tenant B's data only
client_b = AragoraClient(base_url="...", api_key="ara_tenant_b_key")
```

## Error handling

```python
from aragora_sdk import AragoraClient, AuthenticationError, AuthorizationError, RateLimitError

client = AragoraClient.from_env()

try:
    debate = client.debates.create(task="Evaluate options")
except AuthenticationError:
    # 401 - invalid or expired API key
    print("Check your ARAGORA_API_KEY")
except AuthorizationError:
    # 403 - valid key but insufficient permissions
    print("Your API key doesn't have permission for this action")
except RateLimitError as e:
    # 429 - too many requests
    print(f"Rate limited. Retry after {e.retry_after} seconds")
```

## Security best practices

1. **Never commit API keys** to version control. Use environment variables or `.env` files (add `.env` to `.gitignore`).
2. **Use scoped keys** with minimal permissions for production services.
3. **Rotate keys regularly** and revoke unused keys.
4. **Use SSO** for team access instead of shared API keys.
5. **Monitor usage** via the admin dashboard or `aragora auth list-keys` to detect unauthorized access.
