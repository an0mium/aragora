# TypeScript Web Demo

A minimal web app demonstrating the Aragora TypeScript SDK.

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

Then open http://localhost:3000

## Features

- Create and run debates
- Real-time streaming with WebSockets
- View agent rankings
- Run Gauntlet validations

## Project Structure

```
typescript-web/
  package.json       # Dependencies
  index.html         # Entry point
  src/
    main.ts          # Application code
    styles.css       # Styling
```

## Using with React/Next.js

See the [TypeScript Quickstart](../../docs/guides/typescript-quickstart.md) for React integration examples.
