# TypeScript Web Demo

A full-featured web app demonstrating the Aragora TypeScript SDK with real-time streaming, tournaments, and authentication.

## Setup

```bash
# Install dependencies
npm install

# Start the Aragora server (in another terminal)
python -m aragora.server.unified_server --port 8080

# Set API keys (on server side)
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# Start the web app
npm run dev
```

Then open http://localhost:5173

## Features

### Debates
- Create and run multi-agent debates
- Real-time streaming via WebSocket
- Polling fallback mode
- Live event log
- Consensus visualization

### Tournaments
- Create single/double elimination or round robin tournaments
- Track tournament progress
- View final standings

### Rankings
- Agent leaderboard with ELO ratings
- Win/loss tracking
- Win rate calculation

### Authentication
- Login/logout
- API key management
- Token persistence

## Project Structure

```
typescript-web/
  package.json       # Dependencies (aragora-js, vite, typescript)
  index.html         # Entry point with tab navigation
  src/
    main.ts          # Application code
    styles.css       # Styling (dark theme)
```

## SDK Usage Examples

### Create a Debate (Polling)

```typescript
import { AragoraClient } from 'aragora-js';

const client = new AragoraClient({ baseUrl: 'http://localhost:8080' });

const debate = await client.debates.create({
  topic: 'Should we use microservices?',
  agents: ['anthropic-api', 'openai-api'],
  rounds: 3,
  consensus: 'majority',
});

// Poll for completion
while (debate.status !== 'completed') {
  await sleep(2000);
  debate = await client.debates.get(debate.id);
}

console.log(debate.consensus);
```

### Stream a Debate (WebSocket)

```typescript
const ws = client.debates.stream({
  topic: 'Design a rate limiter',
  agents: ['anthropic-api', 'openai-api'],
  rounds: 2,
});

ws.onMessage((event) => {
  switch (event.type) {
    case 'debate_start':
      console.log('Started:', event.debateId);
      break;
    case 'agent_message':
      console.log(`${event.agent}: ${event.content}`);
      break;
    case 'consensus':
      console.log('Consensus:', event.data);
      break;
  }
});

await ws.connect();
```

### Create a Tournament

```typescript
const tournament = await client.tournaments.create({
  name: 'Q1 AI Showdown',
  participants: ['anthropic-api', 'openai-api', 'gemini'],
  format: 'single_elimination',
});

// Wait and get standings
const standings = await client.tournaments.getStandings(tournament.id);
```

### Authentication

```typescript
// Login
const token = await client.auth.login('user@example.com', 'password');
localStorage.setItem('token', token.accessToken);

// Get current user
const user = await client.auth.getCurrentUser();

// Manage API keys
const keys = await client.auth.listApiKeys();
const newKey = await client.auth.createApiKey('CI/CD');
```

## Using with React/Next.js

See the [TypeScript Quickstart](../../docs/guides/typescript-quickstart.md) for React integration examples.

## Build for Production

```bash
npm run build
npm run preview
```

The built files will be in `dist/`.
