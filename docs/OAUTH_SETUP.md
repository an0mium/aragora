# OAuth and SAML Setup Guide

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

### Environment Variables

```bash
# General OAuth settings
ARAGORA_OAUTH_ENABLED=true
ARAGORA_OAUTH_CALLBACK_URL=https://aragora.example.com/api/auth/callback

# Provider-specific (configure as needed)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_TENANT_ID=...

GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
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
   - Authorized redirect URIs: `https://aragora.example.com/api/auth/google/callback`
5. Save the **Client ID** and **Client Secret**

### Step 4: Configure Aragora

```bash
# .env
ARAGORA_OAUTH_ENABLED=true
GOOGLE_CLIENT_ID=123456789-abcdefg.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxx
GOOGLE_OAUTH_CALLBACK=https://aragora.example.com/api/auth/google/callback

# Optional: Restrict to specific domain
GOOGLE_HOSTED_DOMAIN=example.com
```

### Step 5: Test Integration

```bash
# Start server
aragora serve

# OAuth login URL
open "https://aragora.example.com/api/auth/google"

# Should redirect to Google, then back to your callback URL
```

### Google Workspace Restrictions

```bash
# Only allow users from specific domains
GOOGLE_ALLOWED_DOMAINS=example.com,company.org

# Require Google Workspace accounts (no personal Gmail)
GOOGLE_HOSTED_DOMAIN=example.com
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
   - Redirect URI: `https://aragora.example.com/api/auth/azure/callback`

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
# .env
ARAGORA_OAUTH_ENABLED=true
AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_OAUTH_CALLBACK=https://aragora.example.com/api/auth/azure/callback

# For multitenant apps, use 'common' or 'organizations'
# AZURE_TENANT_ID=common
```

### Step 6: Test Integration

```bash
# OAuth login URL
open "https://aragora.example.com/api/auth/azure"
```

### Advanced: Conditional Access

Configure in Azure AD:
1. Go to **Security > Conditional Access**
2. Create policy for Aragora app
3. Require MFA, compliant device, etc.

---

## GitHub OAuth

### Step 1: Register OAuth App

1. Go to GitHub **Settings > Developer settings > OAuth Apps**
2. Click **New OAuth App**
3. Configure:
   - Application name: `Aragora`
   - Homepage URL: `https://aragora.example.com`
   - Authorization callback URL: `https://aragora.example.com/api/auth/github/callback`

### Step 2: Get Credentials

After registration, note:
- Client ID (shown on app page)
- Generate a new Client Secret

### Step 3: Configure Aragora

```bash
# .env
ARAGORA_OAUTH_ENABLED=true
GITHUB_CLIENT_ID=Iv1.xxxxxxxxxxxx
GITHUB_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_OAUTH_CALLBACK=https://aragora.example.com/api/auth/github/callback

# Optional: Restrict to org members
GITHUB_ALLOWED_ORGS=your-org,another-org
```

### Step 4: Organization Restrictions

```bash
# Only allow members of specific organizations
GITHUB_ALLOWED_ORGS=acme-corp,internal-tools

# Require specific team membership
GITHUB_REQUIRED_TEAMS=acme-corp/engineering,acme-corp/platform
```

### GitHub Enterprise

```bash
# For GitHub Enterprise Server
GITHUB_ENTERPRISE_URL=https://github.company.com
GITHUB_API_URL=https://github.company.com/api/v3
```

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
3. Sign-in redirect URIs: `https://aragora.example.com/api/auth/okta/callback`
4. Sign-out redirect URIs: `https://aragora.example.com`
5. Assignments: Configure user/group access

### Step 3: Get Credentials

After creation, note:
- Client ID
- Client Secret
- Okta domain (e.g., `dev-123456.okta.com`)

### Step 4: Configure Aragora

```bash
# .env
ARAGORA_OAUTH_ENABLED=true
OKTA_CLIENT_ID=0oaxxxxxxxxxx
OKTA_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OKTA_DOMAIN=dev-123456.okta.com
OKTA_OAUTH_CALLBACK=https://aragora.example.com/api/auth/okta/callback

# Optional: Use custom authorization server
OKTA_ISSUER=https://dev-123456.okta.com/oauth2/default
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
# Map Okta groups to Aragora roles
OKTA_ADMIN_GROUPS=Aragora-Admins,IT-Admins
OKTA_USER_GROUPS=Aragora-Users,All-Employees
```

---

## SAML 2.0 Configuration

### Generic SAML Setup

Aragora supports any SAML 2.0 compliant Identity Provider.

### Step 1: Get Service Provider Metadata

```bash
# Download SP metadata
curl https://aragora.example.com/api/auth/saml/metadata > sp-metadata.xml
```

Or manually configure with these values:

| Setting | Value |
|---------|-------|
| Entity ID | `https://aragora.example.com/api/auth/saml` |
| ACS URL | `https://aragora.example.com/api/auth/saml/callback` |
| SLO URL | `https://aragora.example.com/api/auth/saml/logout` |
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
# .env
ARAGORA_SAML_ENABLED=true

# Option 1: Metadata URL (recommended)
SAML_IDP_METADATA_URL=https://idp.example.com/metadata

# Option 2: Manual configuration
SAML_IDP_ENTITY_ID=https://idp.example.com
SAML_IDP_SSO_URL=https://idp.example.com/sso
SAML_IDP_SLO_URL=https://idp.example.com/slo
SAML_IDP_CERTIFICATE=/path/to/idp-cert.pem

# SP configuration
SAML_SP_ENTITY_ID=https://aragora.example.com/api/auth/saml
SAML_SP_PRIVATE_KEY=/path/to/sp-key.pem
SAML_SP_CERTIFICATE=/path/to/sp-cert.pem

# Attribute mapping
SAML_ATTR_EMAIL=http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress
SAML_ATTR_FIRST_NAME=http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname
SAML_ATTR_LAST_NAME=http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname
SAML_ATTR_GROUPS=http://schemas.xmlsoap.org/claims/Group
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
# .env
ARAGORA_OIDC_ENABLED=true

# Discovery URL (recommended)
OIDC_ISSUER=https://idp.example.com

# OR manual configuration
OIDC_AUTHORIZATION_URL=https://idp.example.com/authorize
OIDC_TOKEN_URL=https://idp.example.com/token
OIDC_USERINFO_URL=https://idp.example.com/userinfo
OIDC_JWKS_URL=https://idp.example.com/.well-known/jwks.json

# Credentials
OIDC_CLIENT_ID=aragora-client
OIDC_CLIENT_SECRET=xxxxxxxxxxxxx
OIDC_CALLBACK_URL=https://aragora.example.com/api/auth/oidc/callback

# Scopes
OIDC_SCOPES=openid,email,profile

# Attribute mapping
OIDC_CLAIM_EMAIL=email
OIDC_CLAIM_NAME=name
OIDC_CLAIM_GROUPS=groups
```

### Keycloak Configuration

```bash
# Keycloak-specific settings
OIDC_ISSUER=https://keycloak.example.com/realms/aragora
OIDC_CLIENT_ID=aragora
OIDC_CLIENT_SECRET=xxxxxxxxxxxxx

# Enable PKCE (recommended)
OIDC_USE_PKCE=true
```

### Auth0 Configuration

```bash
# Auth0-specific settings
OIDC_ISSUER=https://your-tenant.auth0.com
OIDC_CLIENT_ID=xxxxxxxxxxxxx
OIDC_CLIENT_SECRET=xxxxxxxxxxxxx

# Auth0 audience (API identifier)
OIDC_AUDIENCE=https://aragora.example.com/api
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
echo $GOOGLE_OAUTH_CALLBACK
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

```bash
# .env
ARAGORA_JWT_CLOCK_SKEW=300  # 5 minutes tolerance
```

### Debug Mode

```bash
# Enable OAuth debug logging
export ARAGORA_OAUTH_DEBUG=true
export ARAGORA_LOG_LEVEL=DEBUG

# View OAuth flow details
tail -f logs/aragora.log | grep -i oauth
```

### Testing OAuth Flow

```bash
# Test OAuth authorization URL generation
curl -v "https://aragora.example.com/api/auth/google"

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
# Rotate secrets without downtime:
# 1. Generate new secret in IdP
# 2. Update both secrets in config
GOOGLE_CLIENT_SECRET_OLD=old_secret
GOOGLE_CLIENT_SECRET=new_secret

# 3. Deploy with both secrets active
# 4. After all sessions rotate, remove old secret
```

---

## Related Documentation

- [SECURITY.md](../SECURITY.md) - Security policies and MFA
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Authentication troubleshooting
- [API_REFERENCE.md](API_REFERENCE.md) - Auth API endpoints
