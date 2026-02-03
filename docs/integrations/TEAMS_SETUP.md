# Microsoft Teams Integration Setup Guide

This guide walks you through setting up the Aragora Microsoft Teams bot for your organization.

## Prerequisites

- Aragora server running and accessible via HTTPS
- Microsoft 365 tenant with admin permissions
- Azure Active Directory access for app registration
- Your Aragora server domain (e.g., `aragora.yourcompany.com`)

## Quick Setup (15 minutes)

### Step 1: Register Azure AD Application

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** > **App registrations**
3. Click **New registration**
4. Configure:
   - **Name**: Aragora Teams Bot
   - **Supported account types**: Accounts in any organizational directory
   - **Redirect URI**: Web - `https://YOUR_DOMAIN/api/v1/bots/teams/oauth/callback`
5. Click **Register**
6. Note the **Application (client) ID** - this is your `TEAMS_CLIENT_ID`

### Step 2: Create Client Secret

1. In your app registration, go to **Certificates & secrets**
2. Click **New client secret**
3. Add a description and expiration
4. Click **Add**
5. **Copy the secret value immediately** - this is your `TEAMS_CLIENT_SECRET`

### Step 3: Configure API Permissions

1. Go to **API permissions**
2. Click **Add a permission** > **Microsoft Graph**
3. Add the following **Delegated permissions**:
   - `User.Read`
   - `offline_access`
   - `openid`
   - `profile`
4. Add the following **Application permissions**:
   - `ChannelMessage.Read.Group`
   - `TeamSettings.Read.Group`
5. Click **Grant admin consent** for your organization

### Step 4: Create Bot Registration

1. Go to [Bot Framework Portal](https://dev.botframework.com/bots/new)
2. Or use Azure Bot Service:
   - Search for "Azure Bot" in Azure Portal
   - Create a new Azure Bot resource
   - Link it to your app registration

### Step 5: Configure Aragora Environment

Add the following to your `.env.production` file:

```bash
# Teams Integration
TEAMS_CLIENT_ID=your-app-registration-client-id
TEAMS_CLIENT_SECRET=your-client-secret
TEAMS_BOT_ID=your-bot-id
MICROSOFT_APP_ID=your-app-registration-client-id
MICROSOFT_APP_PASSWORD=your-client-secret

# Optional: Tenant restriction (for single-tenant apps)
# TEAMS_TENANT_ID=your-tenant-id
```

### Step 6: Create Teams App Package

1. Navigate to `deploy/teams/`
2. Edit `manifest.json`:
   - Replace `{{TEAMS_APP_ID}}` with your Application ID
   - Replace `{{TEAMS_BOT_ID}}` with your Bot ID
   - Replace `{{TEAMS_CLIENT_ID}}` with your Client ID
   - Replace `{{ARAGORA_HOST}}` with your server domain
3. Add icon files:
   - `color.png` (192x192 pixels)
   - `outline.png` (32x32 pixels, transparent background)
4. Create ZIP package:
   ```bash
   cd deploy/teams
   zip -r aragora-teams.zip manifest.json color.png outline.png
   ```

### Step 7: Install in Teams

1. Open Microsoft Teams
2. Go to **Apps** > **Manage your apps** > **Upload an app**
3. Choose **Upload a custom app**
4. Select your `aragora-teams.zip` package
5. Click **Add** to install

## Available Commands

| Command | Description |
|---------|-------------|
| `@Aragora debate <topic>` | Start a multi-agent debate |
| `@Aragora plan <topic>` | Debate + implementation plan |
| `@Aragora implement <topic>` | Debate + plan with context snapshot |
| `@Aragora status` | Show active debates |
| `@Aragora leaderboard` | Show agent rankings |
| `@Aragora results <id>` | Get results for a debate |
| `@Aragora help` | Show help message |

## Features

### Adaptive Cards
Debate results are displayed as rich Adaptive Cards with:
- Consensus summary and confidence score
- Agent positions with evidence
- Interactive vote buttons
- Progress indicators during debate

### Threading Support
Replies to @Aragora mentions in a thread will keep the conversation in that thread.

### Compose Extensions
Use the Aragora messaging extension to:
- Start debates from the compose box
- Search past decisions
- Share debate results

### Personal App
The Aragora personal app tab provides:
- Dashboard with debate statistics
- History of past debates
- Quick access to start new debates

## RBAC Integration

Teams users are mapped to Aragora RBAC via Azure AD Object ID:
- `teams:messages:read` - Read messages
- `teams:messages:send` - Send messages
- `teams:debates:create` - Create debates
- `teams:debates:vote` - Vote in debates
- `teams:cards:respond` - Respond to card actions
- `teams:admin` - Admin access

## Troubleshooting

### Bot not responding
1. Verify the messaging endpoint URL is correct:
   `https://YOUR_DOMAIN/api/v1/bots/teams/messages`
2. Check that your server is accessible from Microsoft's IP ranges
3. Review server logs for errors

### "App not responding" error
1. Ensure your bot registration is properly linked to Azure AD app
2. Verify `MICROSOFT_APP_ID` and `MICROSOFT_APP_PASSWORD` are correct
3. Check that the bot endpoint is responding to health checks

### Card actions not working
1. Verify the interactivity endpoint is configured
2. Check that the bot has proper permissions
3. Review the Adaptive Card JSON for errors

### Permission denied errors
1. Verify Azure AD permissions are granted
2. Check RBAC role assignment for the user
3. Ensure tenant authorization is configured

## Security Considerations

### Token Validation
All incoming requests are validated against:
- Azure AD JWT tokens
- Bot Framework authentication
- Request timestamps (to prevent replay attacks)

### Tenant Isolation
For multi-tenant deployments, each tenant's data is isolated using the Azure AD tenant ID.

### Audit Logging
All Teams interactions are logged with:
- User AAD Object ID
- Tenant ID
- Activity type
- Timestamp
- Conversation context

## Advanced Configuration

### Single-Tenant Deployment
For single-tenant apps, add to manifest.json:
```json
"validTenantIds": ["your-tenant-id"]
```

### Custom Branding
Update the app icons and accent color in `manifest.json` to match your organization's branding.

### Rate Limiting
Teams webhook rate limits:
- Messages endpoint: 60 requests/minute
- Status endpoint: 30 requests/minute

Configure in `.env.production`:
```bash
TEAMS_RATE_LIMIT_MESSAGES=60
TEAMS_RATE_LIMIT_STATUS=30
```

## Support

- Documentation: https://docs.aragora.ai/integrations/teams
- Issues: https://github.com/aragora/aragora/issues
- Community: https://aragora.ai/community
