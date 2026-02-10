# Aragora Next.js App Router Template

A starter template for building Aragora-powered applications with Next.js 14+ App Router.

## Features

- Server Components for initial data fetching
- Client Components for interactivity
- Real-time WebSocket streaming for live debate updates
- Modern styling with CSS variables
- TypeScript throughout

## Quick Start

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your Aragora API URL and key

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the app.

## Environment Variables

Create a `.env.local` file:

```bash
# Server-side (private)
ARAGORA_API_URL=http://localhost:8080
ARAGORA_API_KEY=your-api-key

# Client-side (public, exposed to browser)
NEXT_PUBLIC_ARAGORA_API_URL=http://localhost:8080
NEXT_PUBLIC_ARAGORA_API_KEY=your-api-key
```

## Project Structure

```
app/
  layout.tsx          # Root layout with navigation
  page.tsx            # Home page
  globals.css         # Global styles
  debates/
    page.tsx          # Debates list (Server Component)
    new/
      page.tsx        # Create debate form (Client Component)
    [id]/
      page.tsx        # Debate detail (Server Component)
      DebateStream.tsx # Real-time stream (Client Component)

lib/
  aragora.ts          # SDK client configuration

components/           # Reusable components (add as needed)
```

## Key Patterns

### Server Components (Default)

Fetch data on the server for better performance and SEO:

```tsx
// app/debates/page.tsx
import { getServerClient } from '@/lib/aragora';

async function getDebates() {
  const client = getServerClient();
  return client.debates.list();
}

export default async function DebatesPage() {
  const debates = await getDebates();
  return <DebateList debates={debates} />;
}
```

### Client Components

Use for interactivity and browser APIs:

```tsx
// app/debates/new/page.tsx
'use client';

import { useState } from 'react';
import { getClientSideClient } from '@/lib/aragora';

export default function NewDebatePage() {
  const [task, setTask] = useState('');

  const handleSubmit = async () => {
    const client = getClientSideClient();
    await client.debates.create({ task });
  };

  return <form onSubmit={handleSubmit}>...</form>;
}
```

### Real-time Streaming

Subscribe to debate events with WebSockets:

```tsx
'use client';

import { useEffect, useState } from 'react';
import { getClientSideClient } from '@/lib/aragora';

export default function DebateStream({ debateId }) {
  const [events, setEvents] = useState([]);

  useEffect(() => {
    const client = getClientSideClient();
    const stream = client.debates.stream(debateId);

    stream.on('message', (event) => {
      setEvents(prev => [...prev, event]);
    });

    return () => stream.close();
  }, [debateId]);

  return <EventList events={events} />;
}
```

## Learn More

- [Aragora Documentation](https://docs.aragora.ai)
- [Next.js App Router](https://nextjs.org/docs/app)
- [@aragora/sdk](https://www.npmjs.com/package/@aragora/sdk)

## Deployment

Deploy with [Vercel](https://vercel.com):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/an0mium/aragora/tree/main/examples/nextjs-app-router)
