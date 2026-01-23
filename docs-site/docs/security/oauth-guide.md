---
title: OAuth Integration Guide
description: OAuth Integration Guide
---

# OAuth Integration Guide

> **Note:** This guide has been consolidated into [AUTH_GUIDE.md](./authentication).
> See the unified authentication documentation for the most up-to-date information.

This guide covers setting up and using OAuth authentication in Aragora.

## Supported Providers

- **Google OAuth 2.0** (fully implemented)

## Quick Start

### 1. Set Environment Variables

```bash
# Required for Google OAuth
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret

# Callback URL (must match Google Console configuration)
GOOGLE_OAUTH_REDIRECT_URI=http://localhost:8080/api/auth/oauth/google/callback

# Frontend redirect URLs
OAUTH_SUCCESS_URL=http://localhost:3000/auth/callback
OAUTH_ERROR_URL=http://localhost:3000/auth/error

# Security: Allowed redirect hosts (comma-separated)
OAUTH_ALLOWED_REDIRECT_HOSTS=localhost,your-domain.com
```

### 2. Configure Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create or select a project
3. Navigate to **APIs & Services > Credentials**
4. Create **OAuth 2.0 Client ID** (Web application)
5. Add authorized redirect URI: `http://localhost:8080/api/auth/oauth/google/callback`
6. Copy Client ID and Client Secret to environment variables

## OAuth Flow

### Step 1: Initiate Login

Redirect user to start the OAuth flow:

```
GET /api/auth/oauth/google
```

Optional query parameters:
- `redirect_url` - Where to return after login (must be in allowed hosts)

### Step 2: User Consents at Google

User is redirected to Google's consent screen.

### Step 3: Callback Processing

Google redirects to your callback URL with an authorization code.
The server:
1. Validates the CSRF state token
2. Exchanges code for access token
3. Fetches user profile from Google
4. Creates or links user account
5. Generates JWT tokens

### Step 4: Frontend Receives Tokens

User is redirected to `OAUTH_SUCCESS_URL` with tokens in the URL fragment:

```
http://localhost:3000/auth/callback#access_token=xxx&refresh_token=xxx&token_type=Bearer&expires_in=3600&user_id=xxx
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/oauth/google` | GET | Start Google OAuth flow |
| `/api/auth/oauth/google/callback` | GET | Handle OAuth callback |
| `/api/auth/oauth/link` | POST | Link OAuth to existing account |
| `/api/auth/oauth/unlink` | DELETE | Unlink OAuth provider |
| `/api/auth/oauth/providers` | GET | List available providers |

## TypeScript SDK Usage

```typescript
import { AragoraClient } from 'aragora-js';

// Redirect to OAuth
window.location.href = 'http://localhost:8080/api/auth/oauth/google';

// Handle callback in your frontend
const hash = window.location.hash.substring(1);
const params = new URLSearchParams(hash);

const tokens = {
  accessToken: params.get('access_token'),
  refreshToken: params.get('refresh_token'),
  userId: params.get('user_id'),
};

// Use tokens with client
const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  accessToken: tokens.accessToken,
});
```

## Account Linking

Link OAuth to an existing authenticated account:

```typescript
// Must be authenticated first
await client.linkOAuthProvider('google');
```

## Security Features

- **CSRF Protection**: State tokens validated before token exchange
- **Open Redirect Prevention**: Redirect URLs validated against allowlist
- **Token Fragment**: Tokens passed in URL fragment (not query params)
- **Single-Use States**: State tokens consumed after validation
- **Rate Limiting**: 20 requests/minute on OAuth endpoints

## Production Configuration

### Redis State Storage

For multi-instance deployments, use Redis for state storage:

```bash
REDIS_URL=redis://localhost:6379/0
OAUTH_STATE_TTL_SECONDS=600
OAUTH_MAX_STATES=10000
```

### HTTPS Configuration

In production, always use HTTPS:

```bash
GOOGLE_OAUTH_REDIRECT_URI=https://api.your-domain.com/api/auth/oauth/google/callback
OAUTH_SUCCESS_URL=https://app.your-domain.com/auth/callback
OAUTH_ERROR_URL=https://app.your-domain.com/auth/error
OAUTH_ALLOWED_REDIRECT_HOSTS=app.your-domain.com
```

## Troubleshooting

### "Invalid redirect URL"
- Check `OAUTH_ALLOWED_REDIRECT_HOSTS` includes your domain
- Ensure redirect URL matches exactly (including protocol)

### "State token expired"
- State tokens expire after 10 minutes
- User may have taken too long at consent screen
- Retry the OAuth flow

### "OAuth provider not configured"
- Check `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` are set
- Verify environment variables are loaded

## Related Documentation

- [API Reference](../api/reference)
- [Authentication](./overview)
- [Security](./overview)
