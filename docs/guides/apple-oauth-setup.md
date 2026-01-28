# Apple OAuth Setup Guide

This guide explains how to configure Sign in with Apple for Aragora.

## Prerequisites

- Apple Developer Program membership ($99/year)
- Access to [Apple Developer Portal](https://developer.apple.com/)

## Step 1: Create an App ID

1. Go to [Certificates, Identifiers & Profiles](https://developer.apple.com/account/resources/identifiers/list)
2. Click the **+** button to create a new identifier
3. Select **App IDs** and click **Continue**
4. Select **App** as the type
5. Fill in:
   - **Description**: Your app name (e.g., "Aragora Web")
   - **Bundle ID**: Use explicit ID (e.g., `com.yourcompany.aragora`)
6. Under **Capabilities**, enable **Sign in with Apple**
7. Click **Continue** and **Register**

## Step 2: Create a Services ID (for Web)

1. In [Identifiers](https://developer.apple.com/account/resources/identifiers/list), click **+**
2. Select **Services IDs** and click **Continue**
3. Fill in:
   - **Description**: "Aragora Web Sign In"
   - **Identifier**: `com.yourcompany.aragora.web` (this is your `APPLE_OAUTH_CLIENT_ID`)
4. Click **Continue** and **Register**

### Configure the Services ID

1. Find your new Services ID and click on it
2. Enable **Sign in with Apple**
3. Click **Configure**
4. Set up:
   - **Primary App ID**: Select your App ID from Step 1
   - **Domains and Subdomains**: Your domain (e.g., `api.yourcompany.com`)
   - **Return URLs**: Your callback URL (e.g., `https://api.yourcompany.com/api/auth/oauth/apple/callback`)
5. Click **Save** and **Continue**

## Step 3: Create a Private Key

1. Go to [Keys](https://developer.apple.com/account/resources/authkeys/list)
2. Click **+** to create a new key
3. Fill in:
   - **Key Name**: "Aragora Sign In Key"
4. Enable **Sign in with Apple**
5. Click **Configure** and select your Primary App ID
6. Click **Continue** and **Register**
7. **Download the key file** (`.p8`) - this is only available once!
8. Note the **Key ID** (this is your `APPLE_KEY_ID`)

## Step 4: Get Your Team ID

Your Team ID is visible in the top-right corner of the Apple Developer Portal, or in your [Membership Details](https://developer.apple.com/account#MembershipDetailsCard).

## Environment Variables

Set the following environment variables:

```bash
# Required: Services ID created in Step 2
APPLE_OAUTH_CLIENT_ID=com.yourcompany.aragora.web

# Required: Your Apple Developer Team ID
APPLE_TEAM_ID=TEAMID123

# Required: Key ID from Step 3
APPLE_KEY_ID=KEYID456

# Required: Contents of the .p8 file from Step 3
# Either set directly or via file reference
APPLE_PRIVATE_KEY="<contents of AuthKey_KEYID456.p8>"

# Optional: Override redirect URI (auto-configured for localhost in dev)
APPLE_OAUTH_REDIRECT_URI=https://api.yourcompany.com/api/auth/oauth/apple/callback
```

### Private Key Format

The private key can be provided in several ways:

1. **Direct value**: Copy the entire contents of the `.p8` file including header and footer
   ```bash
   export APPLE_PRIVATE_KEY="$(cat AuthKey_KEYID456.p8)"
   ```

2. **From environment**: Reference a secrets manager or vault
   ```bash
   # Use AWS Secrets Manager, HashiCorp Vault, or similar
   APPLE_PRIVATE_KEY=$(aws secretsmanager get-secret-value --secret-id apple-sign-in-key --query SecretString --output text)
   ```

3. **From file** (in your code):
   ```python
   import os
   with open("AuthKey_KEYID456.p8") as f:
       os.environ["APPLE_PRIVATE_KEY"] = f.read()
   ```

## Callback URL Configuration

Apple supports three response modes for the callback:

| Mode | Description | Use Case |
|------|-------------|----------|
| `form_post` | Data sent via POST (default) | Web apps |
| `query` | Data in URL query string | Not recommended (security) |
| `fragment` | Data in URL fragment | Single-page apps |

Aragora defaults to `form_post` which is the most secure option.

## Important Notes

### User Data on First Sign-In Only

Apple only provides the user's name and email on the **first authorization**. On subsequent sign-ins, only the user ID (`sub`) is guaranteed.

**Best Practice**: Store the user's name immediately during the first sign-in flow.

### Private Email Relay

Users can choose to hide their real email using Apple's Private Relay. These emails look like:
```
abc123xyz@privaterelay.appleid.com
```

Messages sent to this address are forwarded to the user's real email. You must configure your domain in the [Apple Developer Portal](https://developer.apple.com/account/resources/services/list) to send emails to relay addresses.

### ID Token Verification

Aragora verifies Apple ID tokens using:
- Apple's public JWKS keys (cached for 24 hours)
- Signature verification (RS256/ES256)
- Claim validation (issuer, audience, expiration)

This prevents token forgery and replay attacks.

## Testing

### Local Development

For local testing, set a development redirect URI:

```bash
APPLE_OAUTH_REDIRECT_URI=http://localhost:8080/api/auth/oauth/apple/callback
```

Note: Apple requires HTTPS for production but allows localhost for testing.

### Verify Configuration

```python
from aragora.server.handlers.oauth_providers.apple import AppleOAuthProvider

provider = AppleOAuthProvider()
print(f"Configured: {provider.is_configured}")

if provider.is_configured:
    url = provider.get_authorization_url(state="test-state")
    print(f"Auth URL: {url}")
```

## Troubleshooting

### "invalid_client" Error

- Verify your Services ID is correct
- Ensure the redirect URI matches exactly (including trailing slashes)
- Check that the private key hasn't expired

### "invalid_grant" Error

- Authorization codes are single-use and expire quickly
- Ensure you're exchanging the code immediately

### "Token signature verification failed"

- The ID token was modified in transit
- Apple's keys may have rotated - the cache will refresh automatically

### Name Not Received

- User data is only sent on first authorization
- User may have denied name sharing
- Check if you stored the name from the initial sign-in

## Security Considerations

1. **Store the private key securely** - Use a secrets manager, not environment files
2. **Validate state parameter** - Prevents CSRF attacks
3. **Use HTTPS in production** - Apple requires it
4. **Store user ID, not email** - The `sub` claim is the stable identifier
5. **Handle private relay emails** - They're legitimate and should be accepted

## API Reference

### Authorization URL

```
GET /api/auth/oauth/apple/login?redirect_uri=<optional>
```

### Callback

```
POST /api/auth/oauth/apple/callback
Content-Type: application/x-www-form-urlencoded

code=<auth_code>&state=<state>&user=<optional_json>
```

### Refresh Token

```
POST /api/auth/oauth/apple/refresh
Content-Type: application/json

{"refresh_token": "<token>"}
```

## See Also

- [Apple Sign In Documentation](https://developer.apple.com/documentation/sign_in_with_apple)
- [Sign in with Apple REST API](https://developer.apple.com/documentation/sign_in_with_apple/sign_in_with_apple_rest_api)
- [Configuring Your Webpage for Sign in with Apple](https://developer.apple.com/documentation/sign_in_with_apple/sign_in_with_apple_js/configuring_your_webpage_for_sign_in_with_apple)
