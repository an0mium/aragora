# Aragora Bot Marketplace Submissions

This directory contains the configuration and assets needed to submit Aragora bots to various app marketplaces.

## Directory Structure

```
marketplace/
├── README.md           # This file
├── slack/
│   ├── manifest.json   # Slack App Manifest (v2)
│   └── assets/         # Icons and screenshots
├── discord/
│   ├── application.json # Discord application config
│   └── assets/         # Icons and screenshots
└── assets/             # Shared brand assets
```

## Slack App Directory Submission

### Prerequisites

1. Slack workspace with admin access
2. Working Aragora API deployment at `api.aragora.ai`
3. SSL certificate configured

### Submission Steps

1. **Create App from Manifest**
   - Go to https://api.slack.com/apps
   - Click "Create New App" > "From an app manifest"
   - Select your workspace
   - Paste contents of `slack/manifest.json`
   - Review and create

2. **Configure Credentials**
   ```bash
   # Add to your environment
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_SIGNING_SECRET=...
   SLACK_APP_TOKEN=xapp-...  # Optional for Socket Mode
   ```

3. **Add App Icons**
   - App icon (512x512 PNG)
   - Workspace icon (96x96 PNG)

4. **Complete App Listing**
   - Go to "App Home" > "Your App's Presence in Slack"
   - Fill in all required fields
   - Add screenshots (1600x900 recommended)

5. **Submit for Review**
   - Go to "Manage Distribution" > "Submit to App Directory"
   - Complete the checklist
   - Submit for review

### Slack App Requirements

- [ ] Privacy policy URL
- [ ] Support URL or email
- [ ] App icon (512x512)
- [ ] At least 3 screenshots
- [ ] Complete app description
- [ ] Working OAuth flow
- [ ] Event subscription verified
- [ ] Slash commands tested

---

## Discord App Directory Submission

### Prerequisites

1. Discord developer account
2. Verified bot (required for 100+ servers)
3. Working interactions endpoint

### Submission Steps

1. **Configure Application**
   - Go to https://discord.com/developers/applications
   - Create or select your application
   - Configure settings per `discord/application.json`

2. **Set Up Bot**
   - Enable "Public Bot" if distributing
   - Configure Privileged Gateway Intents:
     - Message Content Intent (required for mentions)

3. **Register Commands**
   ```bash
   # Use Discord API to register global commands
   curl -X PUT \
     "https://discord.com/api/v10/applications/{APP_ID}/commands" \
     -H "Authorization: Bot {BOT_TOKEN}" \
     -H "Content-Type: application/json" \
     -d @discord/commands.json
   ```

4. **Configure Interactions**
   - Set Interactions Endpoint URL:
     `https://api.aragora.ai/api/bots/discord/interactions`
   - Discord will verify the endpoint

5. **Complete Listing**
   - Go to "App Directory" tab
   - Fill in all required fields
   - Add icon (at least 512x512)
   - Add screenshots

6. **Apply for Verification** (100+ servers)
   - Go to "App Verification"
   - Complete the form
   - Wait for review (2-4 weeks)

### Discord Requirements

- [ ] Privacy policy URL
- [ ] Terms of service URL
- [ ] Support server invite
- [ ] App icon (512x512+)
- [ ] Detailed description
- [ ] Verified interactions endpoint
- [ ] Commands registered globally

---

## Required Assets

### Icons

| Platform | Size | Format | Notes |
|----------|------|--------|-------|
| Slack | 512x512 | PNG | Square, no transparency |
| Slack | 96x96 | PNG | Workspace icon |
| Discord | 512x512+ | PNG/JPG | Can be larger |

### Screenshots

Both platforms require screenshots demonstrating the bot in action:

1. **Debate Start** - Showing the command being used
2. **Debate Progress** - Mid-debate with agent responses
3. **Debate Results** - Final consensus and voting
4. **Gauntlet Results** - Stress-test validation output
5. **Help Command** - Available commands list

Recommended sizes:
- Slack: 1600x900 (16:9)
- Discord: 1280x720 or 1920x1080

---

## Testing Checklist

### Before Submission

- [ ] All slash commands work correctly
- [ ] OAuth flow completes successfully
- [ ] Bot responds within timeout limits
  - Slack: 3 seconds for initial response
  - Discord: 3 seconds for interaction
- [ ] Error messages are user-friendly
- [ ] Rate limiting doesn't affect normal usage
- [ ] Results render correctly on mobile

### Common Issues

1. **Timeout Errors**
   - Respond immediately with acknowledgment
   - Use deferred responses for long operations

2. **Signature Verification**
   - Ensure correct signing secret
   - Check timestamp validation

3. **Permission Errors**
   - Verify all required scopes/permissions
   - Check bot role position in Discord

---

## Environment Variables

```bash
# Slack
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_SIGNING_SECRET=your-secret
SLACK_APP_ID=A0123456789

# Discord
DISCORD_BOT_TOKEN=your-token
DISCORD_APPLICATION_ID=123456789012345678
DISCORD_PUBLIC_KEY=your-public-key
```

---

## Support

- Documentation: https://docs.aragora.ai/integrations
- Issues: https://github.com/aragora-ai/aragora/issues
- Discord: https://discord.gg/aragora
