# aragora.auth

Enterprise authentication module for Aragora. Provides SSO via OIDC and SAML 2.0,
SCIM 2.0 user provisioning, brute-force lockout protection, token rotation,
and admin impersonation with full audit trails.

## Modules

| File | Purpose |
|------|---------|
| `sso.py` | Base SSO provider abstraction (`SSOProvider`, `SSOUser`, `SSOConfig`) |
| `oidc.py` | OpenID Connect provider (Azure AD, Okta, Google, Auth0, Keycloak) |
| `saml.py` | SAML 2.0 SP implementation (requires `python3-saml`) |
| `lockout.py` | Brute-force prevention with exponential backoff (Redis or in-memory) |
| `token_rotation.py` | Automatic token rotation policies and suspicious activity detection |
| `impersonation.py` | Time-limited admin impersonation with audit logging and 2FA |
| `teams_sso.py` | Microsoft Teams SSO via Azure AD token exchange |
| `scim/` | SCIM 2.0 provisioning (RFC 7643/7644) |
| `scim/schemas.py` | User, Group, and Enterprise User resource schemas |
| `scim/server.py` | HTTP endpoints for SCIM user and group CRUD |
| `scim/filters.py` | SCIM filter expression parser (eq, co, sw, and/or/not, etc.) |

## Key Features

- **OIDC** -- OAuth 2.0 / OpenID Connect with PKCE support for major IdPs
- **SAML 2.0** -- Service Provider mode; optional dependency (`python3-saml`)
- **SCIM 2.0** -- Automated user/group provisioning from Okta, Azure AD, OneLogin, JumpCloud, Google
- **Lockout** -- Per-email and per-IP tracking with exponential backoff; Redis-backed or in-memory
- **Token Rotation** -- Usage-based and time-based rotation, IP/UA binding, anomaly detection
- **Impersonation** -- Audited, time-limited admin sessions with 2FA requirement
- **Teams SSO** -- Bot Framework activity-based authentication via Azure AD

## Usage

```python
from aragora.auth import get_sso_provider, OIDCProvider, OIDCConfig

# OIDC authentication
config = OIDCConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    issuer_url="https://login.microsoftonline.com/tenant-id/v2.0",
    callback_url="https://aragora.example.com/auth/callback",
)
provider = OIDCProvider(config)
auth_url = await provider.get_authorization_url(state="random-state")
user = await provider.authenticate(code="auth-code")

# Brute-force lockout
from aragora.auth import get_lockout_tracker

tracker = get_lockout_tracker()
if tracker.is_locked(email=email, ip=client_ip):
    remaining = tracker.get_remaining_time(email=email, ip=client_ip)
    raise Exception(f"Locked for {remaining}s")
tracker.record_failure(email=email, ip=client_ip)

# SCIM provisioning (mount in FastAPI)
from aragora.auth.scim import SCIMServer, SCIMConfig

scim = SCIMServer(SCIMConfig(bearer_token="token", tenant_id="t-123"))
app.include_router(scim.router, prefix="/scim/v2")
```

## Optional Dependencies

| Package | Required For |
|---------|-------------|
| `python3-saml` | SAML 2.0 (`HAS_SAML` flag indicates availability) |
| `redis` | Distributed lockout tracking (falls back to in-memory) |
