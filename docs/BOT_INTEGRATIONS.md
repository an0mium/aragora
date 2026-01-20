# Bot Integrations Guide

> **Last Updated:** 2026-01-20

Aragora is **omnivorous by design**—it communicates bidirectionally across multiple channels. Query from wherever you work; get multi-agent consensus delivered back to you.

**Output Channels**: Web, Slack, Discord, Microsoft Teams, Zoom, WhatsApp Business, Telegram, API.

This guide covers setting up Aragora bots for various chat platforms to enable bidirectional communication - running debates and receiving results directly from your team's chat.

## Table of Contents

- [Overview](#overview)
- [Common Commands](#common-commands)
- [Discord](#discord)
- [Microsoft Teams](#microsoft-teams)
- [Zoom](#zoom)
- [Slack](#slack)
- [WhatsApp Business](#whatsapp-business)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Overview

Aragora bots allow your team to:
- Start multi-agent debates from chat
- Run stress-test validations on decisions
- Check system status
- Vote on debate outcomes
- Receive debate results in threads

All platforms share a common command framework, so the experience is consistent across Discord, Teams, Zoom, and Slack.

---

## Common Commands

| Command | Description | Platforms |
|---------|-------------|-----------|
| `/aragora debate "topic"` | Start a multi-agent debate | All |
| `/aragora gauntlet` | Run adversarial stress-test | Slack, Discord, Teams (with file attachment) |
| `/aragora status` | Check Aragora system status | All |
| `/aragora help` | List available commands | All |

**Example:**
```
/aragora debate "Should we migrate to microservices?"
```

---

## Discord

### Prerequisites

- A Discord server where you have admin permissions
- Access to [Discord Developer Portal](https://discord.com/developers/applications)

### Step 1: Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and name it (e.g., "Aragora")
3. Note the **Application ID** from "General Information"
4. Copy the **Public Key** from "General Information"

### Step 2: Create Bot User

1. Go to "Bot" in the left sidebar
2. Click "Add Bot"
3. Copy the **Bot Token** (keep this secret!)
4. Enable these Privileged Gateway Intents:
   - Message Content Intent

### Step 3: Configure Slash Commands

1. Go to "OAuth2" > "URL Generator"
2. Select scopes: `bot`, `applications.commands`
3. Select bot permissions:
   - Send Messages
   - Send Messages in Threads
   - Embed Links
   - Attach Files
   - Read Message History
   - Use Slash Commands
4. Use the generated URL to invite the bot to your server

### Step 4: Set Up Interactions Endpoint

1. Go to "General Information"
2. Set **Interactions Endpoint URL** to:
   ```
   https://your-api.aragora.ai/api/bots/discord/interactions
   ```
3. Discord will verify the endpoint (requires your server to be running)

### Step 5: Configure Environment

```bash
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_APPLICATION_ID=your-application-id
DISCORD_PUBLIC_KEY=your-public-key
```

### Usage

**Slash Commands:**
```
/debate topic:Should AI be regulated?
/gauntlet statement:Our Q4 projections are accurate
/status
```

**Bot Mention:**
```
@Aragora debate Should we use GraphQL or REST?
```

**Voting:**
After a debate, click the Agree or Disagree button to vote.

---

## Microsoft Teams

### Prerequisites

- Azure subscription
- Admin access to Teams Admin Center
- Access to [Azure Portal](https://portal.azure.com)

### Step 1: Create Azure Bot

1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "Azure Bot" and click Create
3. Fill in:
   - Bot handle: `aragora-bot`
   - Subscription: Your subscription
   - Resource group: Create new or use existing
   - Pricing tier: F0 (free) or S1
   - App ID: Create new Microsoft App ID
4. Click "Review + create"

### Step 2: Configure Bot

1. Go to your bot resource
2. Under "Settings" > "Configuration":
   - Set **Messaging endpoint** to:
     ```
     https://your-api.aragora.ai/api/bots/teams/messages
     ```
3. Under "Channels", add "Microsoft Teams"
4. Note the **App ID** and create a **App Password** (Client Secret)

### Step 3: Create Teams App Manifest

Create `manifest.json`:
```json
{
  "$schema": "https://developer.microsoft.com/en-us/json-schemas/teams/v1.14/MicrosoftTeams.schema.json",
  "manifestVersion": "1.14",
  "version": "1.0.0",
  "id": "your-app-id",
  "packageName": "ai.aragora.teams",
  "developer": {
    "name": "Aragora",
    "websiteUrl": "https://aragora.ai",
    "privacyUrl": "https://aragora.ai/privacy",
    "termsOfUseUrl": "https://aragora.ai/terms"
  },
  "name": {
    "short": "Aragora",
    "full": "Aragora Multi-Agent Debate"
  },
  "description": {
    "short": "Multi-agent debate and validation",
    "full": "Run multi-agent debates and stress-test validations directly in Teams"
  },
  "icons": {
    "color": "color.png",
    "outline": "outline.png"
  },
  "bots": [
    {
      "botId": "your-app-id",
      "scopes": ["team", "personal", "groupchat"],
      "commandLists": [
        {
          "scopes": ["team", "personal", "groupchat"],
          "commands": [
            {
              "title": "debate",
              "description": "Start a multi-agent debate on a topic"
            },
            {
              "title": "gauntlet",
              "description": "Run stress-test validation"
            },
            {
              "title": "status",
              "description": "Check system status"
            },
            {
              "title": "help",
              "description": "Show available commands"
            }
          ]
        }
      ]
    }
  ],
  "permissions": ["identity", "messageTeamMembers"],
  "validDomains": ["aragora.ai", "your-api.aragora.ai"]
}
```

### Step 4: Deploy to Teams

1. Zip `manifest.json` with icon files
2. Go to Teams Admin Center > "Teams apps" > "Manage apps"
3. Upload the app package
4. Approve for your organization

### Step 5: Configure Environment

```bash
TEAMS_APP_ID=your-app-id
TEAMS_APP_PASSWORD=your-client-secret
```

### Usage

**In Teams:**
```
@Aragora debate Should we adopt Kubernetes?
```

**Adaptive Cards:**
Results appear as rich adaptive cards with voting buttons.

---

## Zoom

### Prerequisites

- Zoom account with admin access
- Access to [Zoom Marketplace](https://marketplace.zoom.us)

### Step 1: Create Zoom App

1. Go to [Zoom App Marketplace](https://marketplace.zoom.us/develop/create)
2. Choose "Chatbot" as app type
3. Fill in app information:
   - App Name: Aragora
   - Company Name: Your company
   - Category: Productivity

### Step 2: Configure OAuth

1. Under "App Credentials", note:
   - **Client ID**
   - **Client Secret**
2. Add OAuth Redirect URL:
   ```
   https://your-api.aragora.ai/api/bots/zoom/oauth/callback
   ```

### Step 3: Configure Chatbot

1. Under "Feature" > "Chatbot":
   - Enable chatbot
   - Set **Bot JID** (noted after activation)
   - Add Slash Command: `/aragora`
2. Add Event Subscriptions:
   - Event URL: `https://your-api.aragora.ai/api/bots/zoom/events`
   - Events: `bot_notification`

### Step 4: Security

1. Under "Feature" > "Event Subscription":
   - Note the **Secret Token** for webhook verification
2. Under "App Credentials":
   - Note the **Verification Token**

### Step 5: Configure Environment

```bash
ZOOM_CLIENT_ID=your-client-id
ZOOM_CLIENT_SECRET=your-client-secret
ZOOM_BOT_JID=your-bot-jid
ZOOM_SECRET_TOKEN=your-secret-token
ZOOM_VERIFICATION_TOKEN=your-verification-token
```

### Usage

**In Zoom Chat:**
```
/aragora debate What's the best CI/CD approach?
```

---

## Slack

Slack integration builds on existing Slack handler with enhanced bidirectional features.

### Prerequisites

- Slack workspace admin access
- Existing Slack app or [create new](https://api.slack.com/apps)

### Step 1: Configure Slack App

1. Go to your app settings
2. Under "OAuth & Permissions", add scopes:
   - `chat:write`
   - `commands`
   - `files:read` (for gauntlet attachments)
   - `users:read`
3. Under "Slash Commands", add:
   - `/aragora` with URL: `https://your-api.aragora.ai/api/social/slack/command`

### Step 2: Enable Events

1. Under "Event Subscriptions":
   - Enable events
   - Request URL: `https://your-api.aragora.ai/api/social/slack/events`
   - Subscribe to bot events: `app_mention`, `message.im`

### Step 3: Configure Environment

```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token  # Optional, for Socket Mode
```

### Usage

**Slash Commands:**
```
/aragora debate "Should we use TypeScript or JavaScript?"
/aragora gauntlet  # Attach a file for validation
/aragora status
```

**Bot Mention:**
```
@Aragora help
```

---

## WhatsApp Business

WhatsApp integration enables debate and validation directly in WhatsApp conversations.

### Prerequisites

- WhatsApp Business Account
- Access to [Meta for Developers](https://developers.facebook.com)
- Phone number registered with WhatsApp Business API

### Step 1: Create Meta App

1. Go to [Meta for Developers](https://developers.facebook.com)
2. Create a new app (type: Business)
3. Add WhatsApp product to your app
4. Note your **Phone Number ID** and **WhatsApp Business Account ID**

### Step 2: Configure Webhook

1. Under WhatsApp > Configuration:
   - Webhook URL: `https://your-api.aragora.ai/api/integrations/whatsapp/webhook`
   - Verify Token: Generate a secure random string
   - Subscribe to events:
     - `messages`
     - `message_status`
     - `messaging_postbacks`

### Step 3: Generate Access Token

1. Under WhatsApp > API Setup:
   - Generate a permanent access token (or use temporary for testing)
   - Grant permissions: `whatsapp_business_messaging`, `whatsapp_business_management`

### Step 4: Configure Message Templates

WhatsApp requires pre-approved templates for proactive messages. Create templates for:

1. **Debate Results Template** (`aragora_debate_result`):
   ```
   *Debate Complete*

   Topic: {{1}}
   Consensus: {{2}}
   Confidence: {{3}}%

   View full results: {{4}}
   ```

2. **Gauntlet Alert Template** (`aragora_gauntlet_alert`):
   ```
   *Validation Alert*

   Statement: {{1}}
   Status: {{2}}
   Issues Found: {{3}}

   Details: {{4}}
   ```

Submit templates for review in WhatsApp Manager.

### Step 5: Configure Environment

```bash
WHATSAPP_ACCESS_TOKEN=your-access-token
WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
WHATSAPP_BUSINESS_ACCOUNT_ID=your-business-account-id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your-verify-token
WHATSAPP_PROVIDER=meta  # or 'twilio' for Twilio integration
```

### Alternative: Twilio Integration

For Twilio WhatsApp:

```bash
WHATSAPP_PROVIDER=twilio
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # Twilio sandbox or your number
```

### Usage

**Initiate a Debate:**
```
debate: Should we migrate to cloud?
```

**Run Gauntlet:**
```
gauntlet: Our Q4 revenue will increase 20%
```

**Get Help:**
```
help
```

**Example Interaction:**
```
User: debate: Is remote work better than office work?

Aragora: 🎯 Starting debate...

Topic: "Is remote work better than office work?"
Agents: Claude, GPT-4, Gemini, Grok

I'll send you results when the debate concludes.

---

[After debate completes]

Aragora: ✅ *Debate Complete*

📋 Topic: Is remote work better than office work?

🤝 Consensus: PARTIAL_AGREEMENT
Both have tradeoffs; hybrid model often optimal

📊 Confidence: 78%

🗳️ Would you like to:
1. View full transcript
2. Start a follow-up debate
3. Share results

Reply with 1, 2, or 3
```

### Features

- **Interactive Lists**: Use numbered replies for navigation
- **Rich Messages**: Formatted text with emojis for clarity
- **Status Updates**: Real-time notifications during long debates
- **Threaded Conversations**: Context maintained per conversation
- **Media Support**: Send documents for gauntlet analysis

### Rate Limits

WhatsApp Business API has rate limits:
- 250 messages/second per phone number (tier-based)
- Template messages limited by quality rating
- Session messages unlimited within 24-hour window

---

## Architecture

### Bot Framework

All bots share a common framework:

```
aragora/bots/
├── base.py         # Common classes (BotUser, CommandContext, etc.)
├── commands.py     # Command registry and built-in commands
├── discord_bot.py  # Discord-specific implementation
├── teams_bot.py    # Teams-specific implementation
└── zoom_bot.py     # Zoom-specific implementation
```

### Request Flow

```
Chat Platform → Webhook Handler → Command Registry → Execute → Response
```

1. **Webhook Handler**: Validates signatures, parses platform-specific payload
2. **Command Registry**: Routes to appropriate command handler
3. **Execute**: Runs the command (debate, gauntlet, etc.)
4. **Response**: Formats platform-specific response (embeds, cards, etc.)

### Extending Commands

To add a custom command:

```python
from aragora.bots.commands import get_default_registry, BotCommand
from aragora.bots.base import CommandContext, CommandResult

registry = get_default_registry()

@registry.command(
    name="custom",
    description="My custom command",
    usage="custom <args>",
)
async def handle_custom(ctx: CommandContext) -> CommandResult:
    return CommandResult(
        success=True,
        message=f"Custom response for {ctx.raw_args}",
    )
```

---

## Troubleshooting

### Discord: "Invalid Interaction"

- Verify your Public Key is correct
- Ensure Interactions Endpoint URL is accessible
- Check server logs for signature verification errors

### Teams: Bot Not Responding

- Verify Messaging Endpoint is correct
- Check Azure Bot health in portal
- Ensure Teams channel is enabled

### Zoom: Webhook Verification Failed

- Verify Secret Token matches
- Check timestamp header is present
- Ensure webhook URL is HTTPS

### Slack: Command Timeout

- Slash commands must respond within 3 seconds
- For long operations, respond with acknowledgment then use `response_url`
- Check Slack app Event Subscriptions are enabled

### General: Rate Limiting

- Commands have built-in rate limiting (10/minute per user)
- Debates have additional rate limits based on organization tier
- Check `/api/bots/<platform>/status` for configuration status

---

## Security Considerations

1. **Never commit tokens**: Use environment variables
2. **Verify signatures**: All platforms provide signature verification
3. **Use HTTPS**: All webhook endpoints must be HTTPS in production
4. **Limit scopes**: Request only necessary permissions
5. **Audit logs**: All bot commands are logged for audit

---

## Support

- Report issues: [GitHub Issues](https://github.com/aragora-ai/aragora/issues)
- Documentation: [https://docs.aragora.ai](https://docs.aragora.ai)
- Discord: [Aragora Community](https://discord.gg/aragora)
