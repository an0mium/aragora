---
title: Authentication Guide
description: Authentication Guide
---

# Authentication Guide

Comprehensive guide to authentication in Aragora, covering built-in JWT authentication, OAuth providers, and enterprise SSO.

## Table of Contents

- [Overview](#overview)
- [Authentication Methods](#authentication-methods)
- [Quick Start](#quick-start)
- [JWT Authentication](#jwt-authentication)
- [OAuth 2.0](#oauth-20)
- [Enterprise SSO](#enterprise-sso)
- [Session Management](#session-management)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Aragora supports multiple authentication methods to accommodate different deployment scenarios:

| Method | Protocol | Best For | Complexity |
|--------|----------|----------|------------|
| Built-in JWT | Email/password | Development, small teams | Low |
| Google OAuth | OAuth 2.0 | Google Workspace orgs | Medium |
| Enterprise SSO | OIDC/SAML | Large organizations | High |

### Authentication Flow

```
User → Login Request → Aragora Server → Identity Provider (if SSO)
                              ↓
                       JWT Token Generation
                              ↓
                   Access Token + Refresh Token
                              ↓
                     Session Created in Store
```

---

## Authentication Methods

### Decision Tree

1. **Development/Small Team?** → Use built-in JWT with email/password
2. **Google Workspace users?** → Use Google OAuth
3. **Enterprise with existing IdP?** → Use SSO (OIDC or SAML)
4. **Multiple auth methods needed?** → Enable OAuth + SSO together

---

## Quick Start

### Option 1: Built-in JWT (Simplest)

No additional configuration needed. Users register and login with email/password.

```bash
# Register a new user
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure_password", "name": "User Name"}'

# Login
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure_password"}'
```

### Option 2: Google OAuth

```bash
# .env
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
GOOGLE_OAUTH_REDIRECT_URI=http://localhost:8080/api/auth/oauth/google/callback
OAUTH_SUCCESS_URL=http://localhost:3000/auth/callback
OAUTH_ERROR_URL=http://localhost:3000/auth/error
OAUTH_ALLOWED_REDIRECT_HOSTS=localhost
```

### Option 3: Enterprise SSO (OIDC)

```bash
# .env
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=oidc
ARAGORA_SSO_CLIENT_ID=your-client-id
ARAGORA_SSO_CLIENT_SECRET=your-client-secret
ARAGORA_SSO_ISSUER_URL=https://your-idp.example.com
ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback
```

---

## JWT Authentication

### How It Works

1. User provides email/password or completes OAuth/SSO flow
2. Server validates credentials and generates JWT tokens
3. Access token (short-lived) used for API requests
4. Refresh token (long-lived) used to obtain new access tokens
5. Session tracked in server-side store

### Token Structure

```
Access Token:
- exp: Expiration time (default: 1 hour)
- sub: User ID
- email: User email
- role: User role (user, admin)
- jti: Unique token ID (session ID)

Refresh Token:
- exp: Expiration time (default: 7 days)
- sub: User ID
- jti: Unique token ID
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Create new account |
| `/api/auth/login` | POST | Authenticate and get tokens |
| `/api/auth/logout` | POST | Invalidate current session |
| `/api/auth/refresh` | POST | Get new access token |
| `/api/auth/me` | GET | Get current user profile |

### Configuration

```bash
# JWT Settings
ARAGORA_JWT_SECRET=your-secret-key-min-32-chars
ARAGORA_JWT_EXPIRY_HOURS=1
ARAGORA_JWT_REFRESH_EXPIRY_DAYS=7
ARAGORA_JWT_ALGORITHM=HS256

# Session Limits
ARAGORA_MAX_SESSIONS_PER_USER=10
ARAGORA_SESSION_INACTIVITY_TIMEOUT=86400  # 24 hours
```

### Using Tokens

```python
import requests

# Login
resp = requests.post('http://localhost:8080/api/auth/login', json={
    'email': 'user@example.com',
    'password': 'password'
})
tokens = resp.json()

# Use access token
headers = {'Authorization': f'Bearer {tokens["access_token"]}'}
resp = requests.get('http://localhost:8080/api/debates', headers=headers)

# Refresh when expired
resp = requests.post('http://localhost:8080/api/auth/refresh', json={
    'refresh_token': tokens['refresh_token']
})
new_tokens = resp.json()
```

---

## OAuth 2.0

Aragora's built-in OAuth handler supports **Google OAuth**. For other providers (Azure AD, Okta, GitHub), use the SSO handler with OIDC.

### Google OAuth Setup

#### Step 1: Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create or select a project
3. Navigate to **APIs & Services > Credentials**
4. Create **OAuth 2.0 Client ID** (Web application)
5. Add authorized redirect URI: `https://your-domain.com/api/auth/oauth/google/callback`

#### Step 2: Configure Environment

```bash
GOOGLE_OAUTH_CLIENT_ID=123456789-abc.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxx
GOOGLE_OAUTH_REDIRECT_URI=https://your-domain.com/api/auth/oauth/google/callback
OAUTH_SUCCESS_URL=https://your-domain.com/auth/callback
OAUTH_ERROR_URL=https://your-domain.com/auth/error
OAUTH_ALLOWED_REDIRECT_HOSTS=your-domain.com
```

#### Step 3: Implement Frontend Callback

```typescript
// In your frontend callback page
const hash = window.location.hash.substring(1);
const params = new URLSearchParams(hash);

const tokens = {
  accessToken: params.get('access_token'),
  refreshToken: params.get('refresh_token'),
  userId: params.get('user_id'),
};

// Store tokens and redirect to app
localStorage.setItem('aragora_tokens', JSON.stringify(tokens));
window.location.href = '/dashboard';
```

### OAuth Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/oauth/google` | GET | Start Google OAuth flow |
| `/api/auth/oauth/google/callback` | GET | Handle OAuth callback |
| `/api/auth/oauth/link` | POST | Link OAuth to existing account |
| `/api/auth/oauth/unlink` | DELETE | Unlink OAuth provider |
| `/api/auth/oauth/providers` | GET | List available providers |

### Account Linking

Users can link OAuth providers to existing accounts:

```typescript
// Must be authenticated first
const response = await fetch('/api/auth/oauth/link', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer $\{accessToken\}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    provider: 'google',
    redirect_url: window.location.origin + '/settings'
  })
});

const { auth_url } = await response.json();
window.location.href = auth_url;
```

---

## Enterprise SSO

For enterprise deployments, Aragora supports OIDC and SAML 2.0 protocols with any compatible Identity Provider.

### Supported Identity Providers

| Provider | OIDC | SAML | Notes |
|----------|------|------|-------|
| Azure AD (Entra ID) | Yes | Yes | Recommended: OIDC |
| Okta | Yes | Yes | Recommended: OIDC |
| Google Workspace | Yes | - | Use OIDC |
| OneLogin | Yes | Yes | Both supported |
| PingFederate | Yes | Yes | Both supported |
| ADFS | - | Yes | SAML only |
| Keycloak | Yes | Yes | Both supported |
| Auth0 | Yes | - | OIDC only |

### OIDC Configuration

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
ARAGORA_SSO_SCOPES=openid,email,profile

# Optional: Domain restrictions
ARAGORA_SSO_ALLOWED_DOMAINS=example.com,company.org
```

### Provider-Specific Configuration

#### Azure AD / Entra ID

```bash
ARAGORA_SSO_PROVIDER_TYPE=azure_ad
ARAGORA_SSO_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
ARAGORA_SSO_CLIENT_SECRET=your-secret
ARAGORA_SSO_ISSUER_URL=https://login.microsoftonline.com/YOUR_TENANT_ID/v2.0
```

#### Okta

```bash
ARAGORA_SSO_PROVIDER_TYPE=okta
ARAGORA_SSO_CLIENT_ID=0oaxxxxxxxxxx
ARAGORA_SSO_CLIENT_SECRET=your-secret
ARAGORA_SSO_ISSUER_URL=https://YOUR_DOMAIN.okta.com/oauth2/default
```

#### Keycloak

```bash
ARAGORA_SSO_PROVIDER_TYPE=oidc
ARAGORA_SSO_ISSUER_URL=https://keycloak.example.com/realms/aragora
ARAGORA_SSO_CLIENT_ID=aragora
ARAGORA_SSO_CLIENT_SECRET=your-secret
```

### SAML 2.0 Configuration

```bash
# Enable SAML SSO
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=saml

# SP (Service Provider) Settings
ARAGORA_SSO_ENTITY_ID=https://your-app.example.com/saml/metadata
ARAGORA_SSO_CALLBACK_URL=https://your-app.example.com/auth/sso/callback

# IdP (Identity Provider) Settings
ARAGORA_SSO_IDP_ENTITY_ID=https://idp.example.com/metadata
ARAGORA_SSO_IDP_SSO_URL=https://idp.example.com/sso
ARAGORA_SSO_IDP_CERTIFICATE=/path/to/idp-cert.pem

# Optional: SP certificates for signed requests
ARAGORA_SSO_SP_PRIVATE_KEY=/path/to/sp-key.pem
ARAGORA_SSO_SP_CERTIFICATE=/path/to/sp-cert.pem
```

### SSO Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/sso/login` | GET | Start SSO flow |
| `/auth/sso/callback` | GET/POST | Handle SSO callback |
| `/auth/sso/logout` | GET | Single logout (SLO) |
| `/auth/sso/metadata` | GET | SP metadata (SAML) |

### SAML SP Metadata

Download or configure your IdP with these values:

| Setting | Value |
|---------|-------|
| Entity ID | `https://your-app.example.com/auth/sso` |
| ACS URL | `https://your-app.example.com/auth/sso/callback` |
| SLO URL | `https://your-app.example.com/auth/sso/logout` |
| NameID Format | `urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress` |

---

## Session Management

Aragora provides comprehensive session management. See [SESSION_MANAGEMENT.md](./SESSION_MANAGEMENT.md) for details.

### Key Features

- **Multi-session support**: Users can be logged in from multiple devices
- **Session listing**: View all active sessions
- **Individual revocation**: Revoke specific sessions
- **Activity tracking**: Track last activity per session
- **Device detection**: Identify device type and browser

### Quick Reference

```bash
# List sessions
GET /api/auth/sessions
Authorization: Bearer <access_token>

# Revoke specific session
DELETE /api/auth/sessions/:session_id
Authorization: Bearer <access_token>

# Revoke all other sessions
DELETE /api/auth/sessions
Authorization: Bearer <access_token>
```

---

## Security Best Practices

### Production Checklist

- [ ] Use HTTPS for all endpoints
- [ ] Set strong `ARAGORA_JWT_SECRET` (32+ characters)
- [ ] Store secrets in secure secret manager
- [ ] Configure `OAUTH_ALLOWED_REDIRECT_HOSTS`
- [ ] Enable rate limiting on auth endpoints
- [ ] Set appropriate token expiration times
- [ ] Implement MFA at IdP level (for SSO)
- [ ] Regularly rotate client secrets
- [ ] Monitor for suspicious login patterns
- [ ] Enable audit logging

### Rate Limiting

Auth endpoints are rate limited by default:

| Endpoint | Limit |
|----------|-------|
| `/api/auth/login` | 10/min per IP |
| `/api/auth/register` | 5/min per IP |
| `/api/auth/oauth/*` | 20/min per IP |
| `/auth/sso/*` | 20/min per IP |

### Secret Rotation

```bash
# Rotate JWT secret (will invalidate all sessions)
ARAGORA_JWT_SECRET=new-secret-here

# Rotate OAuth client secret
# 1. Generate new secret in Google Console
# 2. Update environment variable
GOOGLE_OAUTH_CLIENT_SECRET=new_secret
# 3. Redeploy
# 4. Delete old secret in Google Console
```

---

## Troubleshooting

### Common Issues

#### "Invalid redirect URL"
- Check `OAUTH_ALLOWED_REDIRECT_HOSTS` includes your domain
- Verify callback URL matches exactly (including protocol and trailing slash)

#### "State token expired"
- OAuth state tokens expire after 10 minutes
- Retry the authentication flow

#### "Invalid client credentials"
- Verify client ID and secret are correct
- Check for whitespace in environment variables
- Regenerate client secret if needed

#### "User not authorized"
- Check user assignment in IdP (for SSO)
- Verify domain restrictions
- Check group membership requirements

#### Token expires immediately
- Check server clock synchronization
- Verify `ARAGORA_JWT_EXPIRY_HOURS` setting

### Debug Mode

```bash
# Enable auth debug logging
ARAGORA_LOG_LEVEL=DEBUG

# View auth flow details
tail -f logs/aragora.log | grep -iE "(auth|oauth|sso|jwt)"
```

---

## Related Documentation

- [SESSION_MANAGEMENT.md](./SESSION_MANAGEMENT.md) - Detailed session management
- [API_RATE_LIMITS.md](./API_RATE_LIMITS.md) - Rate limiting configuration
- [SECURITY.md](./SECURITY.md) - Security policies
- [API_REFERENCE.md](./API_REFERENCE.md) - Full API documentation
