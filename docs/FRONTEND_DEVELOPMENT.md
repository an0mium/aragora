# Frontend Development Guide

This guide covers setting up and developing the Aragora Live frontend application.

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Location** | `aragora/live/` |
| **Framework** | Next.js 14 (App Router) |
| **Language** | TypeScript |
| **Styling** | Tailwind CSS |
| **State** | React Context + Custom Hooks |
| **Testing** | Jest + RTL + Playwright |
| **Real-time** | WebSocket |

---

## Canonical Frontend

The active, production frontend lives in `aragora/live/` (Next.js App Router).
This is the canonical UI for new feature work and is what powers aragora.ai.

Other frontend-related directories in the repo:
- `aragora-js/`: TypeScript SDK for API consumers (not a UI).
- `frontend/`: legacy prototype with a single hook file; not wired to builds or deployments.

SDK docs: see `aragora-js/README.md`. All frontend feature work should happen in `aragora/live/`.
Route map: see [FRONTEND_ROUTES](./FRONTEND_ROUTES.md) for the full UI surface.

---

## Project Structure

```
aragora/live/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Homepage
│   │   ├── debate/[[...id]]/   # Debate viewer (dynamic route)
│   │   ├── debates/            # Debate list, graph, matrix views
│   │   ├── gauntlet/           # Gauntlet validation UI
│   │   ├── laboratory/         # Agent testing laboratory
│   │   ├── insights/           # Analytics dashboard
│   │   ├── replays/            # Debate replay browser
│   │   ├── auth/               # Login/register
│   │   └── billing/            # Subscription management
│   ├── components/             # React components
│   │   ├── debate-viewer/      # Live/archived debate UI
│   │   ├── deep-audit/         # Gauntlet audit views
│   │   ├── landing/            # Landing page sections
│   │   ├── shared/             # Reusable UI primitives
│   │   ├── auth/               # Auth-related components
│   │   └── billing/            # Billing components
│   ├── hooks/                  # Custom React hooks
│   │   ├── useDebateWebSocket.ts  # Live debate streaming
│   │   ├── useGauntletWebSocket.ts # Gauntlet streaming
│   │   ├── useApi.ts           # API client
│   │   └── useFetch.ts         # Data fetching
│   ├── context/                # React contexts
│   │   ├── AuthContext.tsx     # Authentication state
│   │   └── FeaturesContext.tsx # Feature flags
│   ├── types/                  # TypeScript types
│   │   └── events.ts           # WebSocket event types
│   ├── utils/                  # Utility functions
│   │   ├── supabase.ts         # Supabase client
│   │   ├── sanitize.ts         # HTML sanitization
│   │   └── logger.ts           # Client-side logging
│   └── config.ts               # Configuration
├── __tests__/                  # Component tests
├── e2e/                        # Playwright e2e tests
└── public/                     # Static assets
```

---

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm
- Running Aragora backend (`aragora serve`)

### Setup

```bash
cd aragora/live

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend runs on `http://localhost:3000` by default.

### Environment Variables

Create `.env.local` for local development:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8765/ws

# Optional: Supabase (for debate history)
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Feature flags
NEXT_PUBLIC_ENABLE_STREAMING=true
NEXT_PUBLIC_ENABLE_AUDIENCE=true

# Defaults
NEXT_PUBLIC_DEFAULT_AGENTS=anthropic-api,openai-api,gemini
NEXT_PUBLIC_DEFAULT_ROUNDS=3
```

---

## Development Workflow

### Running Tests

```bash
# Unit/component tests
npm test

# Watch mode
npm run test:watch

# E2E tests (requires running app)
npm run test:e2e

# E2E with UI
npm run test:e2e:ui
```

### Linting

```bash
npm run lint
```

### Building

```bash
# Production build (uses production URLs)
npm run build

# Local build (uses localhost URLs)
npm run build:local
```

---

## Key Patterns

### WebSocket Hooks

The frontend uses custom hooks for real-time WebSocket communication:

```typescript
import { useDebateWebSocket } from '@/hooks/useDebateWebSocket';

function DebateViewer({ debateId }: { debateId: string }) {
  const {
    status,          // 'connecting' | 'streaming' | 'complete' | 'error'
    messages,        // TranscriptMessage[]
    streamingMessages,  // Map<string, StreamingMessage>
    streamEvents,    // StreamEvent[]
    sendVote,        // (choice: string, intensity?: number) => void
    sendSuggestion,  // (suggestion: string) => void
    reconnect,       // Manual reconnect trigger
  } = useDebateWebSocket({
    debateId,
    wsUrl: 'ws://localhost:8765/ws',
    enabled: true,
  });

  // Render debate UI...
}
```

**Features:**
- Automatic reconnection with exponential backoff
- Message deduplication
- Token streaming with sequence ordering
- Orphaned stream cleanup (60s timeout)

### API Client

Use the `useApi` hook for REST API calls:

```typescript
import { useApi } from '@/hooks/useApi';

function LeaderboardPanel() {
  const api = useApi();
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    api.get('/api/leaderboard')
      .then(setAgents)
      .catch(console.error);
  }, []);
}
```

### Authentication Context

```typescript
import { useAuth } from '@/context/AuthContext';

function UserMenu() {
  const { user, signIn, signOut, isLoading } = useAuth();

  if (isLoading) return <LoadingSpinner />;
  if (!user) return <button onClick={signIn}>Sign In</button>;

  return <button onClick={signOut}>Sign Out</button>;
}
```

### Feature Flags

```typescript
import { useFeatures } from '@/hooks/useFeatures';
import { FeatureGuard } from '@/components/FeatureGuard';

// Check programmatically
const { isEnabled } = useFeatures();
if (isEnabled('streaming')) {
  // Show streaming UI
}

// Or use guard component
<FeatureGuard feature="audience">
  <UserParticipation />
</FeatureGuard>
```

---

## Component Conventions

### File Structure

Each component should follow this pattern:

```typescript
// ComponentName.tsx

'use client';  // If using client-side features

import { useState } from 'react';
import { SomeType } from '@/types/events';

interface ComponentNameProps {
  prop1: string;
  prop2?: number;
}

export function ComponentName({ prop1, prop2 = 0 }: ComponentNameProps) {
  // Component logic
  return (
    <div className="...">
      {/* JSX */}
    </div>
  );
}
```

### Styling

Use Tailwind CSS for styling:

```tsx
<div className="bg-black/80 border border-green-500/30 rounded-lg p-4">
  <h2 className="text-green-400 font-mono text-lg">
    Panel Title
  </h2>
  <p className="text-green-300/80 text-sm">
    Content
  </p>
</div>
```

The frontend uses a CRT/terminal aesthetic with:
- Green color palette (`green-400`, `green-500`, `green-300/80`)
- Black backgrounds with opacity
- Monospace fonts (`font-mono`)
- Borders with low opacity (`border-green-500/30`)

### Shared Components

Use components from `src/components/shared/`:

```tsx
import { PanelContainer, PanelHeader, StatusBadge } from '@/components/shared';

function MyPanel() {
  return (
    <PanelContainer>
      <PanelHeader title="My Panel" />
      <StatusBadge status="active" />
    </PanelContainer>
  );
}
```

---

## Testing Patterns

### Component Tests

```typescript
// __tests__/MyComponent.test.tsx

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { MyComponent } from '../src/components/MyComponent';

// Mock dependencies
jest.mock('../src/hooks/useApi', () => ({
  useApi: () => ({
    get: jest.fn().mockResolvedValue({ data: [] }),
  }),
}));

describe('MyComponent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Expected Text')).toBeInTheDocument();
  });

  it('handles user interaction', async () => {
    render(<MyComponent />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button'));
    });

    await waitFor(() => {
      expect(screen.getByText('Updated Text')).toBeInTheDocument();
    });
  });
});
```

### WebSocket Testing

```typescript
// Mock WebSocket for tests
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;

  url: string;
  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
  }

  send = jest.fn();
  close = jest.fn();

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

global.WebSocket = MockWebSocket as any;
```

### E2E Tests (Playwright)

```typescript
// e2e/debates.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Debates', () => {
  test('can view debate list', async ({ page }) => {
    await page.goto('/debates');

    await expect(page.getByRole('heading', { name: /debates/i }))
      .toBeVisible();
  });

  test('can start a new debate', async ({ page }) => {
    await page.goto('/');

    await page.fill('[data-testid="question-input"]', 'Test question');
    await page.click('[data-testid="start-debate"]');

    await expect(page).toHaveURL(/\/debate\//);
  });
});
```

---

## Configuration

### `src/config.ts`

Centralized configuration with environment variable overrides:

```typescript
// API endpoints
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8765/ws';

// Defaults
export const DEFAULT_AGENTS = process.env.NEXT_PUBLIC_DEFAULT_AGENTS || 'anthropic-api,openai-api';
export const DEFAULT_ROUNDS = parseInt(process.env.NEXT_PUBLIC_DEFAULT_ROUNDS || '3', 10);

// Timeouts
export const API_TIMEOUT_MS = 30000;
export const WS_RECONNECT_DELAY_MS = 3000;

// Cache TTLs
export const CACHE_TTL_LEADERBOARD = 5 * 60 * 1000;  // 5 minutes
export const CACHE_TTL_DEBATES = 2 * 60 * 1000;      // 2 minutes

// Feature flags
export const ENABLE_STREAMING = process.env.NEXT_PUBLIC_ENABLE_STREAMING !== 'false';
export const ENABLE_AUDIENCE = process.env.NEXT_PUBLIC_ENABLE_AUDIENCE !== 'false';
```

---

## Adding New Features

### 1. Add a New Page

```bash
# Create page directory
mkdir -p src/app/my-feature

# Create page component
cat > src/app/my-feature/page.tsx << 'EOF'
import { MyFeaturePanel } from '@/components/MyFeaturePanel';

export default function MyFeaturePage() {
  return (
    <main className="min-h-screen bg-black p-8">
      <MyFeaturePanel />
    </main>
  );
}
EOF
```

### 2. Add a New Component

```bash
# Create component
cat > src/components/MyFeaturePanel.tsx << 'EOF'
'use client';

import { useState, useEffect } from 'react';
import { useApi } from '@/hooks/useApi';
import { PanelContainer, PanelHeader } from '@/components/shared';

export function MyFeaturePanel() {
  const api = useApi();
  const [data, setData] = useState(null);

  useEffect(() => {
    api.get('/api/my-feature').then(setData);
  }, []);

  return (
    <PanelContainer>
      <PanelHeader title="My Feature" />
      {/* Content */}
    </PanelContainer>
  );
}
EOF
```

### 3. Add Tests

```bash
# Component test
cat > __tests__/MyFeaturePanel.test.tsx << 'EOF'
import { render, screen } from '@testing-library/react';
import { MyFeaturePanel } from '../src/components/MyFeaturePanel';

jest.mock('../src/hooks/useApi', () => ({
  useApi: () => ({ get: jest.fn().mockResolvedValue({}) }),
}));

describe('MyFeaturePanel', () => {
  it('renders panel header', () => {
    render(<MyFeaturePanel />);
    expect(screen.getByText('My Feature')).toBeInTheDocument();
  });
});
EOF
```

---

## WebSocket Event Types

The frontend handles these WebSocket event types:

| Event Type | Description |
|------------|-------------|
| `debate_start` | Debate initialized with task and agents |
| `debate_end` | Debate completed |
| `agent_message` | Full agent response |
| `token_start` | Token streaming started |
| `token_delta` | Streaming token received |
| `token_end` | Token streaming completed |
| `critique` | Agent critique of another agent |
| `consensus` | Consensus status update |
| `vote` | Agent vote |
| `grounded_verdict` | Citation/evidence verdict |
| `uncertainty_analysis` | Disagreement detection |
| `flip_detected` | Position flip detected |
| `audience_summary` | User participation metrics |

See `src/types/events.ts` for full type definitions.

---

## Debugging

### Enable Debug Logging

The frontend uses a custom logger that respects log levels:

```typescript
import { logger } from '@/utils/logger';

logger.debug('Debug message');  // Only in development
logger.info('Info message');
logger.warn('Warning');
logger.error('Error', error);
```

### WebSocket Debugging

Open browser DevTools and filter Network tab by "WS" to see WebSocket frames.

### React DevTools

Install the React DevTools browser extension for component inspection.

---

## Deployment

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Environment for Production

Set these environment variables in production:

```bash
NEXT_PUBLIC_API_URL=https://api.aragora.ai
NEXT_PUBLIC_WS_URL=wss://api.aragora.ai/ws
NEXT_PUBLIC_SUPABASE_URL=your-production-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-production-key
```

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - CLI and server usage
- [LIBRARY_USAGE.md](LIBRARY_USAGE.md) - Programmatic API
- [API_REFERENCE.md](API_REFERENCE.md) - Backend API endpoints
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
