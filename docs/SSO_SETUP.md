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
# Store PEM values in .env or a secret manager; do not commit private keys.
ARAGORA_SSO_IDP_CERTIFICATE="<PASTE_IDP_CERT_PEM>"

# Optional: SP certificates for signed requests
ARAGORA_SSO_SP_PRIVATE_KEY="<PASTE_SP_PRIVATE_KEY_PEM>"
ARAGORA_SSO_SP_CERTIFICATE="<PASTE_SP_CERT_PEM>"
