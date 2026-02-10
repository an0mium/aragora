# Aragora Remix Template

A starter template for building Aragora-powered applications with Remix.

## Features

- Server-side data loading with loaders
- Form handling with actions
- Progressive enhancement
- TypeScript throughout

## Quick Start

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your Aragora API URL and key

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to see the app.

## Environment Variables

Create a `.env` file:

```bash
ARAGORA_API_URL=http://localhost:8080
ARAGORA_API_KEY=your-api-key
```

## Project Structure

```
app/
  aragora.server.ts      # SDK client (server-only)
  root.tsx               # Root layout
  root.css               # Global styles
  routes/
    _index.tsx           # Home page
    debates._index.tsx   # Debates list
    debates.new.tsx      # Create debate form
    debates.$id.tsx      # Debate detail (add as needed)
```

## Key Patterns

### Data Loading with Loaders

```typescript
// routes/debates._index.tsx
import { json } from '@remix-run/node';
import { getClient } from '~/aragora.server';

export async function loader() {
  const client = getClient();
  const debates = await client.debates.list();
  return json({ debates });
}
```

### Form Actions

```typescript
// routes/debates.new.tsx
import { redirect } from '@remix-run/node';
import { getClient } from '~/aragora.server';

export async function action({ request }) {
  const formData = await request.formData();
  const task = formData.get('task');

  const client = getClient();
  const result = await client.debates.create({ task });

  return redirect(`/debates/${result.debate_id}`);
}
```

### Progressive Enhancement

Remix forms work without JavaScript:

```tsx
import { Form } from '@remix-run/react';

export default function NewDebate() {
  return (
    <Form method="post">
      <input name="task" />
      <button type="submit">Create</button>
    </Form>
  );
}
```

## Learn More

- [Aragora Documentation](https://docs.aragora.ai)
- [Remix Docs](https://remix.run/docs)
- [@aragora/sdk](https://www.npmjs.com/package/@aragora/sdk)
