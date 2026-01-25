# Microsoft Teams App Deployment

This directory contains the Microsoft Teams app manifest and deployment instructions for Aragora.

## Prerequisites

1. **Azure AD App Registration**
   - Create an app registration in Azure Portal
   - Note the Application (client) ID
   - Generate a client secret
   - Configure redirect URIs

2. **Bot Framework Registration**
   - Register a bot in the Azure Bot Service
   - Link it to your Azure AD app registration
   - Configure messaging endpoint

## Configuration

### Environment Variables

Set these in your deployment environment:

```bash
# Azure AD OAuth
TEAMS_CLIENT_ID=<your-azure-ad-client-id>
TEAMS_CLIENT_SECRET=<your-azure-ad-client-secret>
TEAMS_REDIRECT_URI=https://your-domain.com/api/integrations/teams/callback

# Bot configuration
TEAMS_BOT_ID=<your-bot-id>  # Usually same as TEAMS_CLIENT_ID
```

### Manifest Configuration

Before deploying, replace the placeholders in `manifest.json`:

| Placeholder | Description |
|-------------|-------------|
| `{{TEAMS_APP_ID}}` | Unique GUID for your app (generate with `uuidgen`) |
| `{{TEAMS_BOT_ID}}` | Your Azure AD application (client) ID |
| `{{TEAMS_CLIENT_ID}}` | Your Azure AD application (client) ID |
| `{{ARAGORA_HOST}}` | Your Aragora server hostname (e.g., `aragora.yourdomain.com`) |

## Deployment Steps

### 1. Prepare the App Package

```bash
# Generate a unique app ID
export TEAMS_APP_ID=$(uuidgen)

# Replace placeholders in manifest
sed -i "s/{{TEAMS_APP_ID}}/$TEAMS_APP_ID/g" manifest.json
sed -i "s/{{TEAMS_BOT_ID}}/$TEAMS_CLIENT_ID/g" manifest.json
sed -i "s/{{TEAMS_CLIENT_ID}}/$TEAMS_CLIENT_ID/g" manifest.json
sed -i "s/{{ARAGORA_HOST}}/aragora.yourdomain.com/g" manifest.json

# Create the app package
zip -j aragora-teams.zip manifest.json color.png outline.png
```

### 2. Upload to Teams Admin Center

1. Go to [Teams Admin Center](https://admin.teams.microsoft.com/)
2. Navigate to **Teams apps** → **Manage apps**
3. Click **Upload new app**
4. Select `aragora-teams.zip`

### 3. Publish to Organization

For organization-wide deployment:

1. In Teams Admin Center, find "Aragora" in the app list
2. Click the app name
3. Set **Status** to "Allowed"
4. Optionally, create an **App setup policy** to auto-install

### 4. Publish to Teams App Store (Optional)

For public distribution:

1. Go to [Partner Center](https://partner.microsoft.com/)
2. Create a new Teams app submission
3. Upload your app package
4. Complete the certification process

## Azure AD Permissions

The app requires these Microsoft Graph permissions:

| Permission | Type | Description |
|------------|------|-------------|
| `User.Read` | Delegated | Sign in and read user profile |
| `Team.ReadBasic.All` | Delegated | Read team names and descriptions |
| `Channel.ReadBasic.All` | Delegated | Read channel names |
| `ChannelMessage.Send` | Delegated | Send messages to channels |
| `Files.ReadWrite.All` | Delegated | Upload and download files |

Grant admin consent for these permissions in Azure Portal.

## Bot Messaging Endpoint

Configure your bot's messaging endpoint in Azure Bot Service:

```
https://your-domain.com/api/v1/integrations/teams/commands
```

## Icon Requirements

- **color.png**: 192x192 pixels, full color
- **outline.png**: 32x32 pixels, transparent background, single color

## Testing

1. Install the app in a test team
2. Send a message: `@Aragora debate Should we adopt microservices?`
3. Verify the bot responds with an Adaptive Card
4. Check that debates appear in your Aragora dashboard

## Troubleshooting

### Bot not responding
- Check the messaging endpoint is accessible
- Verify bot credentials in Azure Bot Service
- Check Aragora server logs for errors

### OAuth errors
- Verify redirect URI matches exactly
- Check client ID and secret are correct
- Ensure required permissions are granted

### Adaptive Cards not rendering
- Verify Teams client is up to date
- Check card JSON syntax is valid
- Test cards in [Adaptive Cards Designer](https://adaptivecards.io/designer/)

## Support

For issues with the Teams integration:
- GitHub: https://github.com/aragora/aragora/issues
- Documentation: https://docs.aragora.ai/integrations/teams
