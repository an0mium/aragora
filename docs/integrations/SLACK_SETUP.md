# Slack Integration Setup Guide

This guide walks you through setting up the Aragora Slack integration for your workspace.

## Prerequisites

- Aragora server running and accessible via HTTPS
- Slack workspace with admin permissions
- Your Aragora server domain (e.g., `aragora.yourcompany.com`)

## Quick Setup (5 minutes)

### Step 1: Create Slack App

1. Go to [Slack API Apps](https://api.slack.com/apps)
2. Click **Create New App**
3. Choose **From an app manifest**
4. Select your workspace
5. Copy the contents of `deploy/slack/app-manifest.json`
6. Replace all instances of `YOUR_DOMAIN` with your Aragora server domain
7. Click **Create**

### Step 2: Configure Environment

Add the following to your `.env.production` file:

```bash
# Slack Integration
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_CLIENT_ID=your-client-id
SLACK_CLIENT_SECRET=your-client-secret
```

To find these values:
- **Bot Token**: Settings > Install App > Bot User OAuth Token
- **Signing Secret**: Settings > Basic Information > App Credentials > Signing Secret
- **Client ID/Secret**: Settings > Basic Information > App Credentials

### Step 3: Install to Workspace

1. Go to Settings > Install App
2. Click **Install to Workspace**
3. Authorize the requested permissions
4. Verify installation by typing `/aragora help` in any channel

## Available Commands

| Command | Description |
|---------|-------------|
| `/aragora debate <topic>` | Start a multi-agent debate |
| `/aragora plan <topic>` | Debate + implementation plan |
| `/aragora implement <topic>` | Debate + plan with context snapshot |
| `/aragora status` | Show active debates |
| `/aragora vote` | Vote in active debate |
| `/aragora leaderboard` | Show agent rankings |
| `/aragora help` | Show help message |

## Features

### Thread Support
Debates started from a thread will post results back to that thread, keeping conversations organized.

### Rich Block Kit Messages
Debate results include:
- Consensus summary with confidence score
- Agent positions with supporting evidence
- Vote buttons for user participation
- ELO rating updates

### RBAC Integration
Permissions control who can:
- Start debates (`slack.debates.create`)
- Record votes (`slack.votes.record`)
- View status (`slack.commands.read`)

## Troubleshooting

### "Slack signing secret not configured"
Ensure `SLACK_SIGNING_SECRET` is set in your environment and the server has been restarted.

### Commands not responding
1. Check that your server is accessible from the internet
2. Verify the Request URL in your Slack app settings matches your server
3. Check server logs for errors

### Permission denied errors
Contact your Aragora admin to grant the required RBAC permissions for your Slack user.

## Security Considerations

### Signature Verification
All incoming requests are verified using HMAC-SHA256 signature verification. Never disable this in production.

### Workspace Authorization
You can restrict which Slack workspaces can use Aragora by configuring allowed team IDs:

```bash
ARAGORA_SLACK_ALLOWED_TEAMS=T12345678,T87654321
```

### Audit Logging
All Slack commands are logged with:
- User ID and workspace
- Command executed
- Channel context
- Timestamp

## Advanced Configuration

### Multiple Workspaces
For enterprise deployments supporting multiple Slack workspaces:

1. Enable OAuth in your Slack app (Settings > OAuth & Permissions)
2. Set up the OAuth callback URL: `https://YOUR_DOMAIN/api/v1/bots/slack/oauth/callback`
3. Configure workspace-specific settings in the Aragora admin panel

### Custom Commands
To add custom slash commands, extend the handler in:
`aragora/server/handlers/bots/slack/commands.py`

### Proactive Messages
To send messages outside of command responses, use the Slack SDK:

```python
from aragora.integrations.slack import SlackConnector

connector = SlackConnector()
await connector.send_message(
    channel_id="C12345678",
    text="Debate completed!",
    blocks=[...],
    thread_ts="1234567890.123456"  # Optional: reply to thread
)
```

## Support

- Documentation: https://docs.aragora.ai/integrations/slack
- Issues: https://github.com/aragora/aragora/issues
- Community: https://aragora.ai/community
