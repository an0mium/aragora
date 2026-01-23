---
title: OAuth and SAML Setup Guide
description: OAuth and SAML Setup Guide
---

# OAuth and SAML Setup Guide

> **Note:** This guide has been consolidated into [AUTH_GUIDE.md](./authentication).
> See the unified authentication documentation for the most up-to-date information.
> This file is retained for detailed provider-specific configurations.

Configure external identity providers for Aragora authentication.

## Table of Contents

- [Overview](#overview)
- [Google OAuth](#google-oauth)
- [Microsoft Entra ID (Azure AD)](#microsoft-entra-id-azure-ad)
- [GitHub OAuth](#github-oauth)
- [Okta](#okta)
- [SAML 2.0 Configuration](#saml-20-configuration)
- [OpenID Connect (OIDC)](#openid-connect-oidc)
- [Troubleshooting](#troubleshooting)

---

## Overview

Aragora supports multiple authentication methods:

| Method | Protocol | Use Case |
|--------|----------|----------|
| Built-in | Email/password | Small teams, development |
| Google OAuth | OAuth 2.0 | Google Workspace organizations |
| Microsoft Entra ID | OAuth 2.0 / SAML | Enterprise, Microsoft 365 |
| GitHub OAuth | OAuth 2.0 | Developer teams |
| Okta | OAuth 2.0 / SAML | Enterprise SSO |
| Generic SAML | SAML 2.0 | Any SAML-compatible IdP |
| Generic OIDC | OpenID Connect | Any OIDC-compatible IdP |

Note: The built-in OAuth handler currently supports Google only. For Azure AD,
Okta, and other enterprise IdPs, use the SSO handler (OIDC or SAML) with
`ARAGORA_SSO_*` variables.

### Environment Variables

```bash
# Google OAuth (only built-in OAuth provider)
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
GOOGLE_OAUTH_REDIRECT_URI=https://aragora.example.com/api/auth/oauth/google/callback
OAUTH_SUCCESS_URL=https://aragora.example.com/auth/callback
OAUTH_ERROR_URL=https://aragora.example.com/auth/error
OAUTH_ALLOWED_REDIRECT_HOSTS=aragora.example.com,localhost

# SSO (OIDC/SAML) for other providers
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=oidc
ARAGORA_SSO_CLIENT_ID=...
ARAGORA_SSO_CLIENT_SECRET=...
ARAGORA_SSO_ISSUER_URL=https://idp.example.com
ARAGORA_SSO_CALLBACK_URL=https://aragora.example.com/auth/sso/callback
```

---

## Google OAuth

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable the **Google+ API** (or People API)

### Step 2: Configure OAuth Consent Screen

1. Navigate to **APIs & Services > OAuth consent screen**
2. Select **Internal** (for Workspace) or **External**
3. Fill in application details:
   - App name: `Aragora`
   - User support email: Your email
   - Authorized domains: `aragora.example.com`
4. Add scopes:
   - `email`
   - `profile`
   - `openid`

### Step 3: Create OAuth Credentials

1. Navigate to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth client ID**
3. Select **Web application**
4. Configure:
   - Name: `Aragora Production`
   - Authorized JavaScript origins: `https://aragora.example.com`
   - Authorized redirect URIs: `https://aragora.example.com/api/auth/oauth/google/callback`
5. Save the **Client ID** and **Client Secret**

### Step 4: Configure Aragora

```bash
# .env
GOOGLE_OAUTH_CLIENT_ID=123456789-abcdefg.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxx
GOOGLE_OAUTH_REDIRECT_URI=https://aragora.example.com/api/auth/oauth/google/callback
OAUTH_SUCCESS_URL=https://aragora.example.com/auth/callback
OAUTH_ERROR_URL=https://aragora.example.com/auth/error
OAUTH_ALLOWED_REDIRECT_HOSTS=aragora.example.com

# Domain restrictions are not enforced by Google OAuth in Aragora.
# Use IdP-level policies or SSO allowed domains instead.
```

### Step 5: Test Integration

```bash
# Start server
aragora serve

# OAuth login URL
open "https://aragora.example.com/api/auth/oauth/google"

# Should redirect to Google, then back to your callback URL
```

### Google Workspace Restrictions

```bash
# Domain restrictions are not enforced by the Google OAuth handler.
# Enforce at the IdP or use SSO with ARAGORA_SSO_ALLOWED_DOMAINS.
ARAGORA_SSO_ALLOWED_DOMAINS=example.com,company.org
```

---

## Microsoft Entra ID (Azure AD)

### Step 1: Register Application

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID > App registrations**
3. Click **New registration**
4. Configure:
   - Name: `Aragora`
   - Supported account types: Choose based on requirements
     - Single tenant (your org only)
     - Multitenant (any Azure AD)
     - Personal Microsoft accounts
   - Redirect URI: `https://aragora.example.com/auth/sso/callback`

### Step 2: Configure API Permissions

1. Go to **API permissions**
2. Add permissions:
   - Microsoft Graph > Delegated:
     - `email`
     - `openid`
     - `profile`
     - `User.Read`
3. Grant admin consent (if required)

### Step 3: Create Client Secret

1. Go to **Certificates & secrets**
2. Click **New client secret**
3. Set expiration (recommend 24 months)
4. Copy the secret value immediately (shown only once)

### Step 4: Get Configuration Values

From the **Overview** page, note:
- Application (client) ID
- Directory (tenant) ID

### Step 5: Configure Aragora

```bash
# .env (SSO via OIDC)
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=azure_ad
ARAGORA_SSO_ENTITY_ID=https://aragora.example.com/auth/sso
ARAGORA_SSO_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
ARAGORA_SSO_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ARAGORA_SSO_ISSUER_URL=https://login.microsoftonline.com/your-tenant-id/v2.0
ARAGORA_SSO_CALLBACK_URL=https://aragora.example.com/auth/sso/callback

# For multitenant apps, use 'common' or 'organizations'
# ARAGORA_SSO_ISSUER_URL=https://login.microsoftonline.com/common/v2.0
```

### Step 6: Test Integration

```bash
# SSO login URL
open "https://aragora.example.com/auth/sso/login"
```

### Advanced: Conditional Access

Configure in Azure AD:
1. Go to **Security > Conditional Access**
2. Create policy for Aragora app
3. Require MFA, compliant device, etc.

---

## GitHub OAuth

The built-in OAuth handler does not implement GitHub OAuth yet.
If you want GitHub login, use SSO with explicit OIDC endpoints
(GitHub does not expose standard discovery).

```bash
# .env (SSO via OIDC with manual endpoints)
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=github
ARAGORA_SSO_ENTITY_ID=https://aragora.example.com/auth/sso
ARAGORA_SSO_CLIENT_ID=Iv1.xxxxxxxxxxxx
ARAGORA_SSO_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ARAGORA_SSO_AUTH_ENDPOINT=https://github.com/login/oauth/authorize
ARAGORA_SSO_TOKEN_ENDPOINT=https://github.com/login/oauth/access_token
ARAGORA_SSO_USERINFO_ENDPOINT=https://api.github.com/user
ARAGORA_SSO_CALLBACK_URL=https://aragora.example.com/auth/sso/callback
```

Use IdP policies or application-side checks to enforce org or team membership.

---

## Okta

### Step 1: Create Application

1. Log in to Okta Admin Console
2. Go to **Applications > Applications**
3. Click **Create App Integration**
4. Select:
   - Sign-in method: **OIDC - OpenID Connect**
   - Application type: **Web Application**

### Step 2: Configure Application

1. App integration name: `Aragora`
2. Grant type: **Authorization Code**
3. Sign-in redirect URIs: `https://aragora.example.com/auth/sso/callback`
4. Sign-out redirect URIs: `https://aragora.example.com`
5. Assignments: Configure user/group access

### Step 3: Get Credentials

After creation, note:
- Client ID
- Client Secret
- Okta domain (e.g., `dev-123456.okta.com`)

### Step 4: Configure Aragora

```bash
# .env (SSO via OIDC)
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=okta
ARAGORA_SSO_ENTITY_ID=https://aragora.example.com/auth/sso
ARAGORA_SSO_CLIENT_ID=0oaxxxxxxxxxx
ARAGORA_SSO_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ARAGORA_SSO_ISSUER_URL=https://dev-123456.okta.com/oauth2/default
ARAGORA_SSO_CALLBACK_URL=https://aragora.example.com/auth/sso/callback
```

### Step 5: Configure Groups (Optional)

1. In Okta, go to **Applications > Aragora > Sign On**
2. Edit **OpenID Connect ID Token**
3. Add groups claim:
   - Claim name: `groups`
   - Include in: ID Token, Access Token
   - Value type: Groups
   - Filter: Matches regex `.*` (or specific filter)

```bash
# Group-to-role mapping is not exposed via env vars yet.
# Use IdP policies or customize SSO provider code for role mapping.
```

---

## SAML 2.0 Configuration

### Generic SAML Setup

Aragora supports any SAML 2.0 compliant Identity Provider.

### Step 1: Get Service Provider Metadata

```bash
# Download SP metadata
curl https://aragora.example.com/auth/sso/metadata > sp-metadata.xml
```

Or manually configure with these values:

| Setting | Value |
|---------|-------|
| Entity ID | `https://aragora.example.com/auth/sso` |
| ACS URL | `https://aragora.example.com/auth/sso/callback` |
| SLO URL | `https://aragora.example.com/auth/sso/logout` |
| NameID Format | `urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress` |

### Step 2: Configure Identity Provider

Upload the SP metadata to your IdP, or manually configure:
- Entity ID / Audience
- ACS (Assertion Consumer Service) URL
- Required attributes:
  - `email` (required)
  - `firstName` (optional)
  - `lastName` (optional)
  - `groups` (optional, for role mapping)

### Step 3: Get IdP Metadata

Download metadata from your IdP, typically at:
- `/metadata`
- `/saml/metadata`
- `/FederationMetadata/2007-06/FederationMetadata.xml`

### Step 4: Configure Aragora

```bash
# .env (SSO via SAML)
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=saml
ARAGORA_SSO_ENTITY_ID=https://aragora.example.com/auth/sso
ARAGORA_SSO_CALLBACK_URL=https://aragora.example.com/auth/sso/callback

# IdP configuration
ARAGORA_SSO_IDP_ENTITY_ID=https://idp.example.com
ARAGORA_SSO_IDP_SSO_URL=https://idp.example.com/sso
ARAGORA_SSO_IDP_SLO_URL=https://idp.example.com/slo
ARAGORA_SSO_IDP_CERTIFICATE=/path/to/idp-cert.pem

# SP certificates (required if IdP expects signed requests)
ARAGORA_SSO_SP_PRIVATE_KEY=/path/to/sp-key.pem
ARAGORA_SSO_SP_CERTIFICATE=/path/to/sp-cert.pem

# Attribute mapping uses Aragora defaults; customize in code if needed.
```

### Step 5: Generate SP Certificates

```bash
# Generate self-signed certificate for SP
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout sp-key.pem \
  -out sp-cert.pem \
  -subj "/CN=aragora.example.com"
```

### ADFS Configuration

For Active Directory Federation Services:

1. Add Relying Party Trust using SP metadata
2. Configure Claim Rules:
   ```
   Rule 1: Send LDAP Attributes as Claims
   - E-Mail-Addresses -> E-Mail Address
   - Given-Name -> Given Name
   - Surname -> Surname
   - Token-Groups -> Group

   Rule 2: Transform Email to Name ID
   - E-Mail Address -> Name ID (Email format)
   ```

---

## OpenID Connect (OIDC)

### Generic OIDC Configuration

For any OIDC-compliant provider:

```bash
# .env (SSO via OIDC)
ARAGORA_SSO_ENABLED=true
ARAGORA_SSO_PROVIDER_TYPE=oidc
ARAGORA_SSO_ENTITY_ID=https://aragora.example.com/auth/sso
ARAGORA_SSO_CLIENT_ID=aragora-client
ARAGORA_SSO_CLIENT_SECRET=xxxxxxxxxxxxx
ARAGORA_SSO_ISSUER_URL=https://idp.example.com
ARAGORA_SSO_CALLBACK_URL=https://aragora.example.com/auth/sso/callback
ARAGORA_SSO_SCOPES=openid,email,profile

# Optional manual endpoints (override discovery)
ARAGORA_SSO_AUTH_ENDPOINT=https://idp.example.com/authorize
ARAGORA_SSO_TOKEN_ENDPOINT=https://idp.example.com/token
ARAGORA_SSO_USERINFO_ENDPOINT=https://idp.example.com/userinfo
ARAGORA_SSO_JWKS_URI=https://idp.example.com/.well-known/jwks.json
```

### Keycloak Configuration

```bash
# Keycloak-specific settings
ARAGORA_SSO_ISSUER_URL=https://keycloak.example.com/realms/aragora
ARAGORA_SSO_CLIENT_ID=aragora
ARAGORA_SSO_CLIENT_SECRET=xxxxxxxxxxxxx
```

### Auth0 Configuration

```bash
# Auth0-specific settings
ARAGORA_SSO_ISSUER_URL=https://your-tenant.auth0.com
ARAGORA_SSO_CLIENT_ID=xxxxxxxxxxxxx
ARAGORA_SSO_CLIENT_SECRET=xxxxxxxxxxxxx

# Audience / PKCE / claim mapping require code-level changes today.
```

---

## Troubleshooting

### Common Issues

#### "Invalid redirect URI"

**Problem:** OAuth callback URL doesn't match registered URL

**Solution:**
1. Verify exact URL match (including trailing slash)
2. Check protocol (http vs https)
3. Ensure callback is registered in IdP

```bash
# Check configured callback
echo $GOOGLE_OAUTH_REDIRECT_URI
# Must match exactly what's registered in Google Console
```

#### "Invalid client credentials"

**Problem:** Client ID or secret is wrong

**Solution:**
1. Regenerate client secret
2. Check for whitespace in environment variables
3. Ensure correct client ID format

```bash
# Verify credentials are set
env | grep -E "(CLIENT_ID|CLIENT_SECRET)"
```

#### "User not authorized"

**Problem:** User doesn't have access in IdP

**Solution:**
1. Check user assignment in IdP (Azure AD, Okta)
2. Verify domain/organization restrictions
3. Check group membership requirements

#### SAML Signature Validation Failed

**Problem:** SAML response signature doesn't validate

**Solution:**
1. Download fresh IdP certificate
2. Verify certificate hasn't expired
3. Check clock synchronization

```bash
# Check certificate expiry
openssl x509 -in idp-cert.pem -noout -dates

# Check clock
date -u
```

#### Token Expired Immediately

**Problem:** JWT expires immediately after login

**Solution:**
1. Check server clock synchronization
2. Verify IdP and SP clocks are aligned
3. Increase clock skew tolerance

Aragora does not expose a clock-skew override; ensure NTP is configured on
your hosts and review `ARAGORA_JWT_EXPIRY_HOURS` if tokens expire too quickly.

### Debug Mode

```bash
# Enable OAuth debug logging
export ARAGORA_LOG_LEVEL=DEBUG

# View OAuth flow details
tail -f logs/aragora.log | grep -i oauth
```

### Testing OAuth Flow

```bash
# Test OAuth authorization URL generation
curl -v "https://aragora.example.com/api/auth/oauth/google"

# Should return 302 redirect to Google

# After authentication, check callback
# Browser should redirect to your app with authorization code
```

### Verify Token Claims

```python
# Decode JWT to verify claims (Python)
import jwt

token = "eyJ..."
decoded = jwt.decode(token, options={"verify_signature": False})
print(decoded)

# Check for expected claims:
# - sub (subject/user ID)
# - email
# - name
# - groups (if configured)
```

---

## Security Best Practices

### Production Checklist

- [ ] Use HTTPS for all callback URLs
- [ ] Store client secrets in secure secret manager
- [ ] Enable PKCE for public clients
- [ ] Configure token expiration appropriately
- [ ] Implement state parameter validation
- [ ] Enable MFA at IdP level
- [ ] Restrict redirect URIs to specific paths
- [ ] Regularly rotate client secrets
- [ ] Monitor for suspicious login patterns

### Secret Rotation

```bash
# Rotate secrets:
# 1. Generate new secret in IdP
# 2. Update the configured secret
GOOGLE_OAUTH_CLIENT_SECRET=new_secret

# 3. Redeploy and revoke the old secret at the IdP
```

---

## Related Documentation

- [SECURITY.md](./overview) - Security policies and MFA
- [TROUBLESHOOTING.md](../operations/troubleshooting) - Authentication troubleshooting
- [API_REFERENCE.md](../api/reference) - Auth API endpoints
