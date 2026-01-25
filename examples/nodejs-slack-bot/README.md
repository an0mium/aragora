# Aragora Slack Bot

A Slack bot that enables users to run AI debates directly from Slack channels.

## Features

- **Run Debates** - Start multi-agent AI debates from any channel
- **Real-time Updates** - Stream debate progress to Slack threads
- **Agent Rankings** - View the leaderboard of AI agents
- **Tournaments** - Create and track AI tournaments
- **Interactive UI** - Buttons and modals for easy interaction

## Prerequisites

1. A Slack workspace with admin access
2. An Aragora server running
3. Node.js 18+

## Slack App Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" > "From scratch"
3. Name it "Aragora Bot" and select your workspace

### 2. Configure Bot Permissions

Go to "OAuth & Permissions" and add these scopes:

**Bot Token Scopes:**
- `app_mentions:read` - Listen for @mentions
- `chat:write` - Send messages
- `commands` - Handle slash commands
- `users:read` - Get user info

### 3. Create Slash Commands

Go to "Slash Commands" and create:

| Command | Request URL | Description |
|---------|-------------|-------------|
| `/debate` | `https://your-server.com/slack/events` | Start a new AI debate |
| `/rankings` | `https://your-server.com/slack/events` | View agent rankings |
| `/tournament` | `https://your-server.com/slack/events` | Create a tournament |

### 4. Enable Events (Optional)

For app mentions to work:
1. Go to "Event Subscriptions"
2. Enable events
3. Set Request URL to `https://your-server.com/slack/events`
4. Subscribe to `app_mention` under "Bot Events"

### 5. Enable Socket Mode (Development)

For local development without a public URL:
1. Go to "Socket Mode"
2. Enable Socket Mode
3. Generate an app-level token with `connections:write` scope

### 6. Install the App

1. Go to "Install App"
2. Click "Install to Workspace"
3. Copy the Bot User OAuth Token

## Installation

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env

# Edit .env with your tokens
```

## Configuration

Create a `.env` file:

```bash
# Slack credentials
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token  # Only for Socket Mode

# Aragora server
ARAGORA_URL=http://localhost:8080

# Server port
PORT=3000
```

## Running

### Development (Socket Mode)

```bash
# Start with hot reload
npm run dev
```

### Production

```bash
# Build TypeScript
npm run build

# Start server
npm start
```

## Usage

### Start a Debate

```
/debate Should we use microservices or a monolith?
```

The bot will:
1. Post a message with the debate topic
2. Stream agent responses as thread replies
3. Show the consensus when reached
4. Provide buttons for further actions

### View Rankings

```
/rankings
```

Shows the top 10 agents by ELO rating with win/loss records.

### Create a Tournament

```
/tournament Q1 AI Showdown
```

Opens a modal to:
1. Set tournament name
2. Select participating agents
3. Choose format (single/double elimination, round robin)

### Mention the Bot

```
@Aragora Bot what can you do?
```

The bot will reply with available commands.

## Deployment

### Heroku

```bash
heroku create aragora-slack-bot
heroku config:set SLACK_BOT_TOKEN=xoxb-...
heroku config:set SLACK_SIGNING_SECRET=...
heroku config:set ARAGORA_URL=https://your-aragora-server.com
git push heroku main
```

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY dist ./dist
CMD ["node", "dist/index.js"]
```

```bash
docker build -t aragora-slack-bot .
docker run -p 3000:3000 --env-file .env aragora-slack-bot
```

## Architecture

```
slack-bot/
├── src/
│   └── index.ts       # Main bot application
├── package.json       # Dependencies
├── tsconfig.json      # TypeScript config
└── README.md          # This file
```

The bot uses:
- **@slack/bolt** - Slack app framework
- **aragora-js** - Aragora TypeScript SDK
- **tsx** - TypeScript execution for development

## Troubleshooting

### Bot not responding to commands

1. Check that the bot is invited to the channel
2. Verify Request URL is correct in Slack app settings
3. Check server logs for errors

### "dispatch_failed" error

The Slack signing secret may be incorrect. Regenerate it in app settings.

### Socket Mode not connecting

Ensure the app-level token has `connections:write` scope.

## License

MIT
