# Security

This document covers security features implemented in Aragora, including authentication, authorization, sandboxing, and rate limiting.

## Table of Contents

- [Authentication](#authentication)
  - [JWT Tokens](#jwt-tokens)
  - [OAuth 2.0](#oauth-20)
  - [Token Revocation](#token-revocation)
- [Authorization](#authorization)
  - [Role-Based Access](#role-based-access)
  - [Organization Tiers](#organization-tiers)
- [UI Access Controls](#ui-access-controls)
  - [Admin Console](#admin-console)
  - [Developer Portal](#developer-portal)
- [Proof Sandbox](#proof-sandbox)
- [Rate Limiting](#rate-limiting)
- [Security Headers](#security-headers)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)

---

## Authentication

### JWT Tokens

Aragora uses JSON Web Tokens (JWT) for stateless authentication.

#### Token Structure

```python
@dataclass
class JWTPayload:
    sub: str        # User ID
    email: str      # User email
    org_id: str     # Organization ID (optional)
    role: str       # User role
    iat: int        # Issued at (Unix timestamp)
    exp: int        # Expiration (Unix timestamp)
    type: str       # Token type: "access" or "refresh"
    tv: int = 1     # Token version (for logout-all functionality)
```

#### Token Lifecycle

1. **Access Token**: Short-lived (1 hour default), used for API requests
2. **Refresh Token**: Long-lived (30 days default), used to obtain new access tokens

```python
from aragora.billing.jwt_auth import create_token_pair, decode_jwt

# Create tokens after login
tokens = create_token_pair(
    user_id="user-123",
    email="user@example.com",
    org_id="org-456",
    role="member",
)

# Access: tokens.access_token (1 hour)
# Refresh: tokens.refresh_token (30 days)

# Decode and validate
payload = decode_jwt(tokens.access_token)
if payload:
    print(f"User: {payload.sub}")
```

#### Request Authentication

Include the token in the Authorization header:

```http
GET /api/debates HTTP/1.1
Authorization: Bearer <access_token>
```

Or use cookies for browser-based authentication:

```http
Cookie: access_token=<access_token>
```

### OAuth 2.0

Aragora supports OAuth 2.0 authentication with Google (extensible to other providers).

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/oauth/google` | GET | Redirect to Google consent screen |
| `/api/auth/oauth/google/callback` | GET | Handle OAuth callback |
| `/api/auth/oauth/link` | POST | Link OAuth to existing account |
| `/api/auth/oauth/unlink` | DELETE | Unlink OAuth provider |
| `/api/auth/oauth/providers` | GET | List configured providers |

#### OAuth Flow

```
┌─────────┐                ┌─────────┐                ┌─────────┐
│  User   │                │ Aragora │                │ Google  │
└────┬────┘                └────┬────┘                └────┬────┘
     │ 1. Click "Login with Google"                        │
     │────────────────────►│                               │
     │                     │ 2. Generate state, redirect   │
     │                     │──────────────────────────────►│
     │                     │                               │
     │                     │◄──────────────────────────────│
     │ 3. Consent screen   │                               │
     │◄────────────────────│                               │
     │                     │                               │
     │ 4. Approve         │                               │
     │────────────────────────────────────────────────────►│
     │                     │                               │
     │                     │ 5. Callback with code        │
     │                     │◄──────────────────────────────│
     │                     │ 6. Exchange code for tokens   │
     │                     │──────────────────────────────►│
     │                     │◄──────────────────────────────│
     │                     │ 7. Get user info              │
     │                     │──────────────────────────────►│
     │                     │◄──────────────────────────────│
     │ 8. Create/login user, return JWT                    │
     │◄────────────────────│                               │
```

#### Configuration

```bash
# Required for Google OAuth
GOOGLE_OAUTH_CLIENT_ID=your-client-id
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
GOOGLE_OAUTH_REDIRECT_URI=https://yourdomain.com/api/auth/oauth/google/callback

# Frontend URLs
OAUTH_SUCCESS_URL=https://yourdomain.com/auth/callback
OAUTH_ERROR_URL=https://yourdomain.com/auth/error

# Security: Allowed redirect hosts (prevent open redirects)
OAUTH_ALLOWED_REDIRECT_HOSTS=yourdomain.com,localhost
```

#### Security Features

1. **CSRF Protection**: Random state token validated on callback
2. **Open Redirect Prevention**: Redirect URLs validated against allowlist
3. **State Expiration**: OAuth states expire after 10 minutes
4. **Secure Token Delivery**: Tokens passed via URL fragment (not query params)

### Token Revocation

Aragora supports two complementary token revocation mechanisms:

#### 1. Individual Token Blacklisting

Revoke specific tokens before expiration:

```python
from aragora.billing.jwt_auth import (
    get_token_blacklist,
    revoke_token_persistent,
    is_token_revoked_persistent,
)

# In-memory blacklist (single instance)
blacklist = get_token_blacklist()
blacklist.revoke_token(access_token)

# Persistent blacklist (multi-instance)
revoke_token_persistent(access_token)
is_revoked = is_token_revoked_persistent(access_token)
```

#### 2. Token Versioning (Logout All Devices)

Invalidate all tokens for a user at once using token versioning:

```python
# Each user has a token_version field (default: 1)
# Tokens include a 'tv' claim matching the version at creation time
# When token_version is incremented, all existing tokens become invalid

from aragora.storage.user_store import UserStore

user_store = UserStore()

# Invalidate all tokens for a user
new_version = user_store.increment_token_version(user_id)
# All tokens with tv < new_version are now rejected
```

**Token Payload with Version:**

```python
@dataclass
class JWTPayload:
    sub: str        # User ID
    email: str      # User email
    org_id: str     # Organization ID
    role: str       # User role
    iat: float      # Issued at (Unix timestamp)
    exp: float      # Expiration (Unix timestamp)
    tv: int = 1     # Token version (for logout-all)
```

**Validation Flow:**

```
Token Received → Decode JWT → Check Blacklist → Check Token Version
                                                      ↓
                                          Compare tv claim with
                                          user's current token_version
                                                      ↓
                                          Reject if tv < token_version
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/logout` | POST | Revoke current token only |
| `/api/auth/logout-all` | POST | Increment token version, invalidate all sessions |

**Example: Logout from All Devices**

```bash
curl -X POST https://api.aragora.com/api/auth/logout-all \
  -H "Authorization: Bearer <access_token>"
```

Response:
```json
{
  "message": "All sessions terminated",
  "sessions_invalidated": true,
  "token_version": 2
}
```

#### Blacklist Storage

- **In-Memory**: Default, single-instance deployments
- **SQLite**: Persistent, survives restarts
- **Redis**: Multi-instance deployments (planned)

#### Security Considerations

1. **Rate Limiting**: `/api/auth/logout-all` is rate-limited to 3 requests/minute to prevent abuse
2. **Immediate Effect**: Current token is also blacklisted for immediate revocation
3. **No Token Enumeration**: Token version only increments, never exposes active session count

---

## Authorization

### Role-Based Access

| Role | Permissions |
|------|-------------|
| `viewer` | Read debates, view leaderboards |
| `member` | Create debates, vote, participate |
| `admin` | Manage organization, users, billing |
| `owner` | Full access, delete organization |

```python
from aragora.billing.jwt_auth import extract_user_from_request

# In handler
auth_ctx = extract_user_from_request(handler, user_store)
if not auth_ctx.is_authenticated:
    return error_response("Not authenticated", 401)

if auth_ctx.role not in ["admin", "owner"]:
    return error_response("Insufficient permissions", 403)
```

### Organization Tiers

Tiers determine rate limits and feature access:

| Tier | Rate Limit | Burst | Features |
|------|------------|-------|----------|
| `free` | 10/min | 60 | Basic debates |
| `starter` | 50/min | 100 | + Priority support |
| `professional` | 200/min | 400 | + API access |
| `enterprise` | 1000/min | 2000 | + Custom features |

---

## UI Access Controls

### Admin Console

The admin console (`/admin`) surfaces system health, circuit breakers, recent errors, and rate limits.
It must be restricted to admin or owner roles.

Security requirements:
- JWT auth enabled and validated on `/api/system/*` handlers.
- Role enforcement: only `admin` or `owner` should access admin endpoints.
- Prefer additional controls (SSO, allowlist, VPN) for production.

### Developer Portal

The developer portal (`/developer`) allows users to manage API keys and view usage stats.

Security requirements:
- Authenticated user with a valid access token.
- API key issuance is restricted to the requesting user.
- API keys are bearer credentials; display once and encourage immediate secure storage.

---

## Proof Sandbox

Formal verification code runs in a secure sandbox:

### Resource Limits

```python
from aragora.verification.sandbox import ProofSandbox, SandboxConfig

sandbox = ProofSandbox(
    timeout=30.0,           # Max execution time (seconds)
    memory_mb=512,          # Memory limit (MB)
    max_output_bytes=1024*1024,  # Output truncation (1MB)
)
```

### Isolation Features

| Feature | Implementation |
|---------|----------------|
| **Process Isolation** | `start_new_session=True` for clean process group |
| **Memory Limit** | `RLIMIT_AS` (address space limit) |
| **CPU Time Limit** | `RLIMIT_CPU` (backup timeout) |
| **File Descriptors** | `RLIMIT_NOFILE` (256 max) |
| **Process Count** | `RLIMIT_NPROC` (64 max) |
| **Network Disabled** | `no_proxy=*` environment |
| **Restricted PATH** | `/usr/local/bin:/usr/bin:/bin` only |
| **Temp Cleanup** | Automatic directory removal |

### Timeout Enforcement

```python
# Hard kill on timeout
try:
    stdout, stderr = await asyncio.wait_for(
        process.communicate(),
        timeout=config.timeout_seconds,
    )
except asyncio.TimeoutError:
    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
```

---

## Rate Limiting

See [RATE_LIMITING.md](./RATE_LIMITING.md) for detailed documentation.

### Quick Overview

```python
from aragora.server.middleware.rate_limit import (
    rate_limit,
    get_rate_limiter,
    RateLimitResult,
)

# Decorator usage
@rate_limit(requests_per_minute=30)
def handle_debate_create(self, handler):
    ...

# Manual check
limiter = get_rate_limiter()
result = limiter.allow(client_ip="/api/debates")
if not result.allowed:
    return error_response("Rate limit exceeded", 429)
```

### Rate Limit Types

1. **Per-IP**: Default, by client IP address
2. **Per-User**: Authenticated user ID
3. **Per-Endpoint**: Specific endpoint limits
4. **Per-Tier**: Based on subscription tier

---

## Security Headers

Recommended headers for production:

```python
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}
```

### CORS Configuration

```bash
# Allowed origins for cross-origin requests
ARAGORA_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

---

## Environment Variables

### Authentication

| Variable | Description | Required |
|----------|-------------|----------|
| `JWT_SECRET_KEY` | Secret for signing JWTs | Yes (production) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Access token TTL | No (default: 60) |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token TTL | No (default: 30) |

### OAuth

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_OAUTH_CLIENT_ID` | Google OAuth client ID | For OAuth |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth client secret | For OAuth |
| `GOOGLE_OAUTH_REDIRECT_URI` | Callback URL | For OAuth |
| `OAUTH_ALLOWED_REDIRECT_HOSTS` | Allowed redirect domains | Recommended |

### Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_RATE_LIMIT` | Default requests/min | 60 |
| `ARAGORA_IP_RATE_LIMIT` | Per-IP requests/min | 120 |
| `ARAGORA_BURST_MULTIPLIER` | Burst capacity factor | 2.0 |

---

## Best Practices

### Production Checklist

1. **Use HTTPS**: Always use TLS in production
2. **Set JWT_SECRET_KEY**: Use a strong, random secret
3. **Configure CORS**: Restrict to known origins
4. **Enable Rate Limiting**: Prevent abuse
5. **Configure OAuth Allowlist**: Prevent open redirects
6. **Use Persistent Blacklist**: For token revocation across restarts
7. **Monitor Logs**: Watch for authentication failures
8. **Rotate Secrets**: Periodically rotate JWT secrets

### Secure Deployment

```bash
# Production environment
export JWT_SECRET_KEY=$(openssl rand -base64 32)
export ARAGORA_ALLOWED_ORIGINS=https://yourdomain.com
export OAUTH_ALLOWED_REDIRECT_HOSTS=yourdomain.com
export ARAGORA_RATE_LIMIT=60
```

### Logging

Security events are logged with structured format:

```
INFO  oauth_login: user@example.com via Google
WARN  oauth_redirect_blocked: host=evil.com not in allowlist
INFO  token_revoked jti=abc123...
WARN  rate_limit_exceeded for 1.2.3.4 on debate_create
```

---

## See Also

- [Rate Limiting](./RATE_LIMITING.md) - Detailed rate limiting docs
- [Formal Verification](./FORMAL_VERIFICATION.md) - Sandbox security details
- [Environment](./ENVIRONMENT.md) - All environment variables
- [API Reference](./API_REFERENCE.md) - Authentication headers
