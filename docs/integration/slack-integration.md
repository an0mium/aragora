# Slack Integration Guide

This guide covers setting up Aragora's Slack integration for multi-workspace support with OAuth-based app installation.

## Overview

Aragora's Slack integration enables:
- Slash commands (`/aragora debate`, `/aragora gauntlet`, `/aragora status`, `/aragora agents`)
- Interactive components (buttons, menus)
- Multi-workspace support via OAuth
- Automatic token management and encryption
- Thread-based debates (keeps channels clean)
- **Decision Receipts** - Cryptographically signed audit trails with PDF export
- Rate limiting (30 requests/minute per workspace)
- SSRF protection on all webhook responses

## Prerequisites

- Slack workspace admin access
- Aragora server running with public URL (for OAuth callbacks)
- PostgreSQL or SQLite database for workspace storage

## Quick Start

### 1. Create Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From scratch**
3. Name it "Aragora" and select your workspace
4. Go to **OAuth & Permissions** → **Scopes** → **Bot Token Scopes**

### 2. Required OAuth Scopes

Add these scopes to your Slack app:

| Scope | Purpose |
|-------|---------|
| `channels:history` | Read channel messages for evidence collection |
| `channels:read` | List channels for workspace discovery |
| `chat:write` | Post messages and debate results |
| `commands` | Register slash commands |
| `team:read` | Get workspace info during OAuth |
| `users:read` | Resolve user info for mentions |

### 3. Configure Slash Commands

Go to **Slash Commands** and create:

| Command | Request URL | Description |
|---------|-------------|-------------|
| `/aragora` | `https://your-domain/api/v1/integrations/slack/commands` | Main Aragora command |

### 4. Enable Events (Optional)

For event subscriptions:

1. Go to **Event Subscriptions** → Enable Events
2. Request URL: `https://your-domain/api/v1/integrations/slack/events`
3. Subscribe to bot events: `message.channels`, `app_home_opened`

### 5. Configure Interactive Components

Go to **Interactivity & Shortcuts**:

1. Enable Interactivity
2. Request URL: `https://your-domain/api/v1/integrations/slack/interactive`

### 6. Set Up OAuth

Go to **OAuth & Permissions**:

1. Add Redirect URL: `https://your-domain/api/integrations/slack/callback`
2. Copy your **Client ID** and **Client Secret**

## Environment Variables

Set these in your Aragora server:

```bash
# Required for OAuth
SLACK_CLIENT_ID=your-client-id
SLACK_CLIENT_SECRET=your-client-secret
SLACK_REDIRECT_URI=https://your-domain/api/integrations/slack/callback

# Required for webhook verification
SLACK_SIGNING_SECRET=your-signing-secret

# Optional: Default bot token (for single-workspace mode)
SLACK_BOT_TOKEN=xoxb-your-bot-token

# Optional: Custom OAuth scopes
SLACK_SCOPES=channels:history,chat:write,commands,users:read,team:read,channels:read

# Recommended: Token encryption
ARAGORA_ENCRYPTION_KEY=your-32-byte-encryption-key
```

## Installation Flow

### For Workspace Admins

1. Visit `https://your-domain/api/integrations/slack/install`
2. Click "Allow" to authorize Aragora
3. You'll be redirected back with a success message
4. The bot token is securely stored for your workspace

### Programmatic Installation

```bash
# Initiate OAuth flow (opens browser)
curl "https://your-domain/api/integrations/slack/install?tenant_id=your-tenant"

# After OAuth callback, workspace is stored automatically
```

## API Endpoints

### OAuth Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/integrations/slack/install` | GET | Redirect to Slack OAuth |
| `/api/integrations/slack/callback` | GET | OAuth callback handler |
| `/api/integrations/slack/uninstall` | POST | Handle app uninstall |

### Command Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/integrations/slack/commands` | POST | Slash command handler |
| `/api/v1/integrations/slack/interactive` | POST | Interactive component handler |
| `/api/v1/integrations/slack/events` | POST | Event subscription handler |
| `/api/v1/integrations/slack/status` | GET | Integration status |

## Slash Command Usage

```
/aragora debate "Should we adopt microservices?"
/aragora gauntlet "Review this API design"
/aragora help
```

### Command Options

```
/aragora debate <topic>        Start a multi-agent debate on a topic
/aragora gauntlet <statement>  Run adversarial stress-testing
/aragora status [debate_id]    Check debate status or view recent debates
/aragora agents                List available AI agents and their ELO ratings
/aragora help                  Show available commands
```

### Debate Results

When a debate completes, Aragora posts:
- Consensus status and confidence level
- Final conclusion with supporting arguments
- Participant agents and rounds used
- **Decision Receipt URL** - Cryptographically signed audit trail

Example response:
```
Debate Complete: Should we adopt microservices?

✓ Consensus: Yes (82% confidence)
★★★★☆ Rounds: 3 | Participants: claude, gpt-4

Conclusion: Based on the debate, microservices architecture is
recommended for better scalability and independent deployments...

📜 Receipt: https://your-domain/receipts/receipt-abc123
```

### Thread-Based Responses

Debates run in threads to keep channels clean:
1. Initial response acknowledges the command
2. Progress updates posted to thread
3. Final result appears in thread with channel notification

## Multi-Workspace Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Workspace A   │     │   Workspace B   │     │   Workspace C   │
│   (Team ABC)    │     │   (Team XYZ)    │     │   (Team 123)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  Aragora Server │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ Workspace Store │
                        │ (Encrypted)     │
                        └─────────────────┘
```

Each workspace has:
- Isolated OAuth tokens (encrypted at rest)
- Separate rate limits
- Optional tenant association

## Security Best Practices

### Token Encryption

Always set `ARAGORA_ENCRYPTION_KEY` in production:

```bash
# Generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"

# Set in environment
export ARAGORA_ENCRYPTION_KEY=your-generated-key
```

### Webhook Verification

Aragora verifies all Slack webhooks using HMAC-SHA256:

1. Set `SLACK_SIGNING_SECRET` from your app's Basic Information page
2. All unverified requests are rejected with 401

### HTTPS Required

OAuth redirects require HTTPS in production. For local development:
- Use ngrok or similar tunnel
- Set `SLACK_REDIRECT_URI` to your tunnel URL

### Rate Limiting

Slash commands are rate-limited to protect both Aragora and Slack API limits:
- **30 requests per minute** per workspace
- Exceeded limits return a friendly retry message
- Rate limits reset every minute

### SSRF Protection

Response URLs are validated to prevent Server-Side Request Forgery:
- Only `hooks.slack.com` and `api.slack.com` domains allowed
- HTTPS required (HTTP blocked)
- Localhost and internal IPs blocked
- Validates URL scheme, host, and path patterns

## Troubleshooting

### "Invalid Signature" Errors

1. Verify `SLACK_SIGNING_SECRET` matches your app
2. Check server clock synchronization
3. Ensure request body isn't modified by middleware

### OAuth Callback Fails

1. Verify redirect URI matches exactly (including trailing slash)
2. Check `SLACK_CLIENT_ID` and `SLACK_CLIENT_SECRET`
3. Review server logs for detailed error

### Bot Not Responding

1. Check workspace is active: `GET /api/v1/integrations/slack/status`
2. Verify bot has required scopes
3. Check rate limits haven't been exceeded

### Token Decryption Fails

1. Ensure `ARAGORA_ENCRYPTION_KEY` hasn't changed
2. Install cryptography package: `pip install cryptography`
3. Re-install app to get fresh token

## Database Schema

The workspace store uses this schema:

```sql
CREATE TABLE slack_workspaces (
    workspace_id TEXT PRIMARY KEY,    -- Slack team_id
    workspace_name TEXT NOT NULL,
    access_token TEXT NOT NULL,       -- Encrypted xoxb-* token
    bot_user_id TEXT NOT NULL,
    installed_at REAL NOT NULL,
    installed_by TEXT,                -- User who installed
    scopes TEXT,                      -- Comma-separated scopes
    tenant_id TEXT,                   -- Link to Aragora tenant
    is_active INTEGER DEFAULT 1
);
```

## Decision Receipts

Every debate generates a Decision Receipt - a cryptographically signed audit trail:

### Features
- **Unique ID**: Each receipt has a UUID for tracking
- **SHA-256 Integrity**: Content-addressable hashing prevents tampering
- **Multiple Formats**: JSON, HTML, Markdown, PDF
- **Verification**: Cryptographic signature validation

### Accessing Receipts

From Slack:
1. Click the receipt URL in the debate result message
2. View in browser or download as PDF

Via API:
```bash
# Get receipt as JSON
curl https://your-domain/api/v2/receipts/{receipt_id}

# Get as PDF
curl https://your-domain/api/v2/receipts/{receipt_id}?format=pdf -o receipt.pdf

# Verify integrity
curl https://your-domain/api/v2/receipts/{receipt_id}/verify
```

### Receipt Contents
- Debate ID, topic, and timestamp
- Participating agents and their contributions
- Consensus status and confidence score
- Final conclusion with reasoning
- Integrity hash for verification

## See Also

- [Microsoft Teams Integration](./teams-integration.md)
- [Enterprise Connectors](./enterprise-connectors.md)
- [OAuth Setup Guide](../OAUTH_SETUP.md)
- [Decision Receipts API](../API_REFERENCE.md#receipts)
