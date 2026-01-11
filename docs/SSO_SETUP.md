# SSO/Enterprise Authentication Setup

This guide covers configuring Single Sign-On (SSO) for enterprise authentication in Aragora.

## Overview

Aragora supports two SSO protocols:
- **OIDC (OpenID Connect)** - Modern OAuth 2.0-based authentication
- **SAML 2.0** - Enterprise XML-based authentication

Supported Identity Providers (IdPs):
- Azure Active Directory (Azure AD / Entra ID)
- Okta
- Google Workspace
- OneLogin
- PingFederate
- Any OIDC or SAML 2.0 compliant IdP

## Prerequisites

### Required Packages

```bash
# For OIDC (recommended for most use cases)
pip install PyJWT httpx

# For SAML 2.0 (required for production SAML)
pip install python3-saml

# Or install all SSO dependencies
pip install aragora[sso]
```

### Production Requirements

1. **HTTPS Required**: All SSO callback URLs must use HTTPS in production
2. **SAML Signature Validation**: The `python3-saml` library is required for production SAML deployments
3. **Secure Key Storage**: Never commit certificates or secrets to version control

## Quick Start

### 1. Choose Your Protocol

| Use Case | Recommended Protocol |
|----------|---------------------|
| Modern web apps | OIDC |
| Enterprise with existing SAML IdP | SAML |
| Azure AD / Microsoft 365 | OIDC (with Azure AD) |
| Google Workspace | OIDC (with Google) |
| Okta | Either (OIDC preferred) |

### 2. Basic OIDC Configuration

```bash
# Enable SSO
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=oidc

# OIDC Settings
ARAGORA_SSO_CLIENT_ID=your-client-id
ARAGORA_SSO_CLIENT_SECRET=your-client-secret
ARAGORA_SSO_ISSUER_URL=https://your-idp.example.com
ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback
ARAGORA_SSO_ENTITY_ID=https://your-app.example.com

# Optional: Restrict to specific domains
ARAGORA_SSO_ALLOWED_DOMAINS=example.com,company.org
```

### 3. Basic SAML Configuration

```bash
# Enable SSO
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=saml

# SP (Service Provider) Settings
ARAGORA_SSO_ENTITY_ID=https://your-app.example.com/saml/metadata
ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback

# IdP (Identity Provider) Settings
ARAGORA_SSO_IDP_ENTITY_ID=https://idp.example.com/metadata
ARAGORA_SSO_IDP_SSO_URL=https://idp.example.com/sso
ARAGORA_SSO_IDP_CERTIFICATE="-----BEGIN CERTIFICATE-----
MIICpDCCAYwCCQD...
-----END CERTIFICATE-----"

# Optional: SP certificates for signed requests
ARAGORA_SSO_SP_CERTIFICATE="-----BEGIN CERTIFICATE-----..."
ARAGORA_SSO_SP_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----..."
```

## Provider-Specific Setup

### Azure AD (Entra ID)

1. **Register Application**
   - Go to Azure Portal > Azure Active Directory > App registrations
   - Click "New registration"
   - Name: "Aragora"
   - Supported account types: Choose based on your needs
   - Redirect URI: `https://your-app.example.com/auth/sso/callback`

2. **Configure Application**
   - Under "Certificates & secrets", create a new client secret
   - Under "API permissions", add:
     - `openid`
     - `email`
     - `profile`

3. **Environment Variables**
   ```bash
   ARAGORA_SSO_ENABLED=true
   ARAGORA_SSO_PROVIDER_TYPE=azure_ad
   ARAGORA_SSO_CLIENT_ID=<Application (client) ID>
   ARAGORA_SSO_CLIENT_SECRET=<Client secret value>
   ARAGORA_SSO_ISSUER_URL=https://login.microsoftonline.com/<Tenant ID>/v2.0
   ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback
   ```

### Okta

1. **Create Application**
   - Go to Okta Admin Console > Applications > Create App Integration
   - Sign-in method: OIDC - OpenID Connect
   - Application type: Web Application

2. **Configure Application**
   - Sign-in redirect URIs: `https://your-app.example.com/auth/sso/callback`
   - Sign-out redirect URIs: `https://your-app.example.com`
   - Assignments: Assign users/groups

3. **Environment Variables**
   ```bash
   ARAGORA_SSO_ENABLED=true
   ARAGORA_SSO_PROVIDER_TYPE=okta
   ARAGORA_SSO_CLIENT_ID=<Client ID>
   ARAGORA_SSO_CLIENT_SECRET=<Client Secret>
   ARAGORA_SSO_ISSUER_URL=https://your-org.okta.com
   ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback
   ```

### Google Workspace

1. **Create OAuth Client**
   - Go to Google Cloud Console > APIs & Services > Credentials
   - Create OAuth client ID
   - Application type: Web application
   - Authorized redirect URIs: `https://your-app.example.com/auth/sso/callback`

2. **Environment Variables**
   ```bash
   ARAGORA_SSO_ENABLED=true
   ARAGORA_SSO_PROVIDER_TYPE=google
   ARAGORA_SSO_CLIENT_ID=<Client ID>.apps.googleusercontent.com
   ARAGORA_SSO_CLIENT_SECRET=<Client Secret>
   ARAGORA_SSO_ISSUER_URL=https://accounts.google.com
   ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback

   # Restrict to your Google Workspace domain
   ARAGORA_SSO_ALLOWED_DOMAINS=your-company.com
   ```

## Advanced Configuration

### Role Mapping

Map IdP roles/groups to Aragora roles:

```python
from aragora.auth import SSOConfig

config = SSOConfig(
    # ... other settings ...
    role_mapping={
        "IdP_Admin_Group": "admin",
        "IdP_Moderator_Group": "moderator",
        "IdP_User_Group": "user",
    },
    default_role="user",
)
```

### Domain Restrictions

Limit access to specific email domains:

```bash
# Allow only these domains
ARAGORA_SSO_ALLOWED_DOMAINS=company.com,trusted-partner.org

# Note: This is enforced server-side after IdP authentication
```

### Session Duration

Control how long SSO sessions last:

```bash
# Session duration in seconds (default: 8 hours = 28800)
ARAGORA_SSO_SESSION_DURATION=28800

# Range: 300 (5 min) to 604800 (7 days)
```

### Auto-Provisioning

Automatically create user accounts on first login:

```bash
# Enable (default)
ARAGORA_SSO_AUTO_PROVISION=true

# Disable to require pre-existing accounts
ARAGORA_SSO_AUTO_PROVISION=false
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/sso/login` | GET/POST | Initiate SSO login flow |
| `/auth/sso/callback` | GET/POST | Handle IdP callback |
| `/auth/sso/logout` | GET/POST | Logout and revoke session |
| `/auth/sso/metadata` | GET | SAML SP metadata (SAML only) |
| `/auth/sso/status` | GET | SSO configuration status |

### Login Flow

```bash
# Initiate login (returns redirect to IdP)
curl -L "https://your-app.example.com/auth/sso/login"

# Or get auth URL as JSON
curl -H "Accept: application/json" \
  "https://your-app.example.com/auth/sso/login"
```

Response:
```json
{
  "auth_url": "https://idp.example.com/authorize?...",
  "state": "abc123...",
  "provider": "oidc"
}
```

### Check Status

```bash
curl "https://your-app.example.com/auth/sso/status"
```

Response:
```json
{
  "enabled": true,
  "configured": true,
  "provider": "oidc",
  "entity_id": "https://your-app.example.com",
  "callback_url": "https://your-app.example.com/auth/sso/callback",
  "auto_provision": true,
  "allowed_domains": ["company.com"]
}
```

## Security Best Practices

### Production Checklist

1. **Use HTTPS everywhere**
   - Callback URLs must use HTTPS
   - All SSO endpoints should be HTTPS-only

2. **Install proper SAML library**
   ```bash
   # Required for production SAML
   pip install python3-saml
   ```

3. **Validate certificates**
   - Ensure IdP certificates are from trusted sources
   - Rotate certificates before expiry

4. **Set environment mode**
   ```bash
   ARAGORA_ENV=production
   ```
   This enforces:
   - HTTPS callback validation
   - SAML signature validation
   - Secure defaults

5. **Restrict domains**
   ```bash
   ARAGORA_SSO_ALLOWED_DOMAINS=your-company.com
   ```

6. **Monitor authentication events**
   - Check server logs for SSO errors
   - Set up alerts for authentication failures

### Certificate Management

For SAML with signed requests:

```bash
# Generate SP certificate (self-signed for testing)
openssl req -x509 -newkey rsa:2048 \
  -keyout sp-key.pem -out sp-cert.pem \
  -days 365 -nodes \
  -subj "/CN=Aragora SP"

# Set in environment
ARAGORA_SSO_SP_CERTIFICATE="$(cat sp-cert.pem)"
ARAGORA_SSO_SP_PRIVATE_KEY="$(cat sp-key.pem)"
```

## Troubleshooting

### Common Issues

#### "SSO not configured"

Check that:
1. `ARAGORA_SSO_ENABLED=true` is set
2. Required provider settings are configured
3. Server has been restarted after config changes

#### "Invalid or expired state parameter"

Causes:
- Session timeout during login
- Browser cookies disabled
- Multiple login attempts

Solution: Clear cookies and try again

#### "Domain not allowed"

The user's email domain is not in `ARAGORA_SSO_ALLOWED_DOMAINS`.

#### "SSO callback URL must use HTTPS"

In production mode, callback URLs must use HTTPS. Check:
```bash
ARAGORA_SSO_CALLBACK_URL=https://...  # Must start with https://
```

#### "python3-saml required for production SAML"

Install the SAML library:
```bash
pip install python3-saml
```

Or set development mode:
```bash
ARAGORA_ENV=development  # Not recommended for production
```

### Debug Mode

Enable debug logging:

```bash
ARAGORA_LOG_LEVEL=DEBUG
ARAGORA_DEBUG=true
```

Check logs for detailed SSO flow information.

## Programmatic Usage

### Python API

```python
from aragora.auth import get_sso_provider, SSOConfig

# Get configured provider
provider = get_sso_provider()

# Or create custom provider
from aragora.auth.oidc import OIDCProvider, OIDCConfig

config = OIDCConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    issuer_url="https://your-idp.example.com",
    callback_url="https://your-app.example.com/auth/sso/callback",
)

provider = OIDCProvider(config)

# Generate login URL
auth_url = await provider.get_authorization_url(state="random-state")

# Authenticate user from callback
user = await provider.authenticate(code="auth-code-from-callback")
print(f"Authenticated: {user.email}")
```

### Custom Handler Integration

```python
from aragora.server.handlers.sso import SSOHandler

# Register with your server
handler = SSOHandler()
server.add_handler(handler)
```

## Environment Variable Reference

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ARAGORA_SSO_ENABLED` | No | Enable SSO authentication | `false` |
| `ARAGORA_SSO_PROVIDER_TYPE` | Yes (if enabled) | `oidc`, `saml`, `azure_ad`, `okta`, `google` | `oidc` |
| `ARAGORA_SSO_CLIENT_ID` | OIDC | OAuth client ID | - |
| `ARAGORA_SSO_CLIENT_SECRET` | OIDC | OAuth client secret | - |
| `ARAGORA_SSO_ISSUER_URL` | OIDC | OIDC issuer URL | - |
| `ARAGORA_SSO_CALLBACK_URL` | Yes | Callback URL for auth response | - |
| `ARAGORA_SSO_ENTITY_ID` | Yes | Service provider entity ID | - |
| `ARAGORA_SSO_IDP_ENTITY_ID` | SAML | IdP entity ID | - |
| `ARAGORA_SSO_IDP_SSO_URL` | SAML | IdP SSO URL | - |
| `ARAGORA_SSO_IDP_SLO_URL` | SAML (optional) | IdP logout URL | - |
| `ARAGORA_SSO_IDP_CERTIFICATE` | SAML | IdP X.509 certificate (PEM) | - |
| `ARAGORA_SSO_SP_CERTIFICATE` | SAML (optional) | SP X.509 certificate (PEM) | - |
| `ARAGORA_SSO_SP_PRIVATE_KEY` | SAML (optional) | SP private key (PEM) | - |
| `ARAGORA_SSO_ALLOWED_DOMAINS` | No | Comma-separated allowed email domains | - |
| `ARAGORA_SSO_AUTO_PROVISION` | No | Auto-create users on first login | `true` |
| `ARAGORA_SSO_SESSION_DURATION` | No | Session duration in seconds | `28800` (8h) |

---

See also:
- [ENVIRONMENT.md](./ENVIRONMENT.md) - Full environment variable reference
- [SECURITY.md](./SECURITY.md) - Security best practices
- [API_REFERENCE.md](./API_REFERENCE.md) - API documentation
