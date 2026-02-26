# Frontend Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the landing page debate results, sharing infrastructure, and Oracle page visually compelling and shareable.

**Architecture:** Three phases building on each other: (1) Replace flat proposals list with perspective card grid, (2) Add public permalinks with dynamic OG images and share toolbar, (3) Refine Oracle visual layer with canvas particles, smoother tentacles, and luminescent text.

**Tech Stack:** Next.js 16, React 19, Tailwind CSS 4, `@vercel/og` (ImageResponse), HTML Canvas API

---

### Task 1: Create PerspectiveCard Component

**Files:**
- Create: `aragora/live/src/components/PerspectiveCard.tsx`
- Test: visual inspection (component is pure presentational)

**Step 1: Create the PerspectiveCard component**

```tsx
'use client';

// Role-to-color mapping for the 6 landing page analyst roles
const ROLE_COLORS: Record<string, { border: string; text: string; glow: string }> = {
  'strategic analyst': { border: 'var(--acid-cyan)', text: 'var(--acid-cyan)', glow: 'rgba(0, 255, 255, 0.08)' },
  'devil\'s advocate': { border: 'var(--crimson)', text: 'var(--crimson)', glow: 'rgba(255, 0, 64, 0.08)' },
  'implementation expert': { border: 'var(--acid-green)', text: 'var(--acid-green)', glow: 'rgba(57, 255, 20, 0.08)' },
  'industry analyst': { border: 'var(--purple)', text: 'var(--purple)', glow: 'rgba(191, 0, 255, 0.08)' },
  'risk assessor': { border: 'var(--gold, #ffd700)', text: 'var(--gold, #ffd700)', glow: 'rgba(255, 215, 0, 0.08)' },
  'synthesizer': { border: 'var(--acid-magenta)', text: 'var(--acid-magenta)', glow: 'rgba(255, 0, 255, 0.08)' },
};

function getRoleStyle(agentName: string) {
  const lower = agentName.toLowerCase();
  for (const [key, style] of Object.entries(ROLE_COLORS)) {
    if (lower.includes(key)) return style;
  }
  // Fallback: try matching legacy agent names
  if (lower.includes('analyst') || lower.includes('supportive')) return ROLE_COLORS['strategic analyst'];
  if (lower.includes('critic') || lower.includes('critical')) return ROLE_COLORS['devil\'s advocate'];
  if (lower.includes('balanced') || lower.includes('moderator')) return ROLE_COLORS['implementation expert'];
  if (lower.includes('contrarian')) return ROLE_COLORS['risk assessor'];
  if (lower.includes('synthesizer')) return ROLE_COLORS['synthesizer'];
  return { border: 'var(--acid-cyan)', text: 'var(--acid-cyan)', glow: 'rgba(0, 255, 255, 0.08)' };
}

// Extract a short role label from the agent name
function getRoleLabel(agentName: string): string {
  const lower = agentName.toLowerCase();
  for (const key of Object.keys(ROLE_COLORS)) {
    if (lower.includes(key)) return key.split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
  }
  // Fallback labels for legacy agent names
  if (lower.includes('analyst') || lower.includes('supportive')) return 'Strategic Analyst';
  if (lower.includes('critic') || lower.includes('critical')) return 'Devil\'s Advocate';
  if (lower.includes('balanced') || lower.includes('moderator')) return 'Implementation Expert';
  if (lower.includes('contrarian')) return 'Risk Assessor';
  if (lower.includes('synthesizer')) return 'Synthesizer';
  return agentName;
}

interface PerspectiveCardProps {
  agentName: string;
  content: string;
  isFullWidth?: boolean;
}

export function PerspectiveCard({ agentName, content, isFullWidth }: PerspectiveCardProps) {
  const style = getRoleStyle(agentName);
  const label = getRoleLabel(agentName);

  return (
    <div
      className={`border border-[var(--border)] bg-[var(--surface)] p-4 transition-all duration-200 hover:shadow-lg ${
        isFullWidth ? 'col-span-full' : ''
      }`}
      style={{
        borderLeftWidth: '3px',
        borderLeftColor: style.border,
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.boxShadow = `0 0 20px ${style.glow}`;
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.boxShadow = 'none';
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span
          className="inline-block w-2 h-2 rounded-full"
          style={{ backgroundColor: style.border }}
        />
        <span className="text-xs font-bold font-mono uppercase tracking-wider" style={{ color: style.text }}>
          {label}
        </span>
        <span className="text-[10px] font-mono text-[var(--text-muted)] opacity-60">
          {agentName !== label.toLowerCase() ? agentName : ''}
        </span>
      </div>
      <p className="text-sm text-[var(--text)] whitespace-pre-wrap leading-relaxed font-mono">
        {content}
      </p>
    </div>
  );
}
```

**Step 2: Verify the file compiles**

Run: `cd aragora/live && npx tsc --noEmit src/components/PerspectiveCard.tsx 2>&1 | head -20`
Expected: no errors (or only unrelated global type issues)

**Step 3: Commit**

```bash
git add aragora/live/src/components/PerspectiveCard.tsx
git commit -m "feat(live): add PerspectiveCard component for debate result display"
```

---

### Task 2: Refactor DebateResultPreview with Card Grid

**Files:**
- Modify: `aragora/live/src/components/DebateResultPreview.tsx`

**Step 1: Replace the proposals section with a PerspectiveCard grid and add verdict badge**

Replace the entire `DebateResultPreview` component. Key changes:
- Import `PerspectiveCard`
- Replace the flat proposals `<div>` (lines 128-145) with a 2-column grid of `PerspectiveCard` components
- Detect the synthesizer agent and render it full-width at the bottom
- Add verdict badge to the summary bar (color-coded by verdict type)
- Make critiques and votes collapsible (collapsed by default)
- Keep receipt section unchanged

The summary bar gets a verdict badge:
```tsx
const VERDICT_COLORS: Record<string, string> = {
  approved: 'bg-[var(--acid-green)] text-[var(--bg)]',
  approved_with_conditions: 'bg-[var(--gold,#ffd700)] text-[var(--bg)]',
  needs_review: 'bg-[var(--warning,#ff8c00)] text-[var(--bg)]',
  rejected: 'bg-[var(--crimson)] text-white',
};
```

The proposals section becomes:
```tsx
{/* Perspectives Grid */}
<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  {Object.entries(result.proposals)
    .filter(([agent]) => !agent.toLowerCase().includes('synthesizer'))
    .map(([agent, content]) => (
      <PerspectiveCard key={agent} agentName={agent} content={content} />
    ))}
  {/* Synthesizer spans full width */}
  {Object.entries(result.proposals)
    .filter(([agent]) => agent.toLowerCase().includes('synthesizer'))
    .map(([agent, content]) => (
      <PerspectiveCard key={agent} agentName={agent} content={content} isFullWidth />
    ))}
</div>
```

Critiques and votes become collapsible:
```tsx
const [showDetails, setShowDetails] = useState(false);
// ...
<button onClick={() => setShowDetails(!showDetails)} className="...">
  {showDetails ? '[-] Hide critique details' : '[+] Show critique details'}
</button>
{showDetails && (
  <>
    {/* existing critiques and votes JSX */}
  </>
)}
```

**Step 2: Verify the landing page renders correctly**

Run: `cd aragora/live && npx tsc --noEmit 2>&1 | head -20`
Expected: no type errors

**Step 3: Commit**

```bash
git add aragora/live/src/components/DebateResultPreview.tsx
git commit -m "feat(live): replace flat proposals with perspective card grid layout"
```

---

### Task 3: Create OG Image API Route

**Files:**
- Create: `aragora/live/src/app/api/og/[debateId]/route.tsx`

**Step 1: Create the OG image generation endpoint**

```tsx
import { ImageResponse } from 'next/og';
import { NextRequest } from 'next/server';

export const runtime = 'edge';

const VERDICT_COLORS: Record<string, string> = {
  approved: '#39ff14',
  approved_with_conditions: '#ffd700',
  needs_review: '#ff8c00',
  rejected: '#ff0040',
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ debateId: string }> }
) {
  const { debateId } = await params;

  // Fetch debate data from the backend API
  // Use the API_URL from environment or fall back to production
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

  let topic = 'AI Multi-Model Debate';
  let verdict = 'completed';
  let confidence = 0;
  let participantCount = 0;
  let roundsUsed = 0;
  let dissentCount = 0;

  try {
    const res = await fetch(`${apiBase}/api/v1/debates/${debateId}`, {
      next: { revalidate: 3600 }, // Cache for 1 hour
    });
    if (res.ok) {
      const data = await res.json();
      topic = data.topic || data.question || topic;
      verdict = data.verdict || (data.consensus_reached ? 'approved' : 'needs_review');
      confidence = Math.round((data.confidence || 0) * 100);
      participantCount = data.participants?.length || 0;
      roundsUsed = data.rounds_used || 0;
      dissentCount = data.dissenting_views?.length || 0;
    }
  } catch {
    // Use defaults if backend is unreachable
  }

  // Truncate topic for display
  const displayTopic = topic.length > 120 ? topic.slice(0, 117) + '...' : topic;
  const verdictLabel = verdict.replace(/_/g, ' ').toUpperCase();
  const verdictColor = VERDICT_COLORS[verdict] || '#00ffff';

  return new ImageResponse(
    (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          height: '100%',
          backgroundColor: '#0a0a0a',
          padding: '60px',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
          <span style={{ color: '#39ff14', fontSize: '24px', fontWeight: 'bold', letterSpacing: '4px' }}>
            ARAGORA
          </span>
          <span style={{ color: '#666', fontSize: '16px' }}>
            Multi-AI Verdict
          </span>
        </div>

        {/* Topic */}
        <div style={{ display: 'flex', flex: 1, flexDirection: 'column', justifyContent: 'center' }}>
          <p style={{ color: '#e0e0e0', fontSize: '36px', lineHeight: 1.3, marginBottom: '40px', fontWeight: 600 }}>
            &ldquo;{displayTopic}&rdquo;
          </p>

          {/* Verdict bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '20px',
              padding: '20px 28px',
              borderRadius: '12px',
              border: `2px solid ${verdictColor}40`,
              backgroundColor: `${verdictColor}10`,
              marginBottom: '30px',
            }}
          >
            <span style={{ color: verdictColor, fontSize: '20px', fontWeight: 'bold', letterSpacing: '2px' }}>
              {verdictLabel}
            </span>
            {confidence > 0 && (
              <span style={{ color: '#9a9a9a', fontSize: '20px' }}>
                {confidence}% confidence
              </span>
            )}
          </div>
        </div>

        {/* Footer stats */}
        <div style={{ display: 'flex', gap: '24px', color: '#666', fontSize: '18px' }}>
          {participantCount > 0 && <span>{participantCount} AI analysts</span>}
          {roundsUsed > 0 && <span>{roundsUsed} rounds</span>}
          {dissentCount > 0 && <span>{dissentCount} dissenting views</span>}
          <span style={{ marginLeft: 'auto' }}>aragora.ai</span>
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  );
}
```

**Step 2: Verify the route compiles**

Run: `cd aragora/live && npx tsc --noEmit 2>&1 | head -20`
Expected: no type errors

**Step 3: Commit**

```bash
git add aragora/live/src/app/api/og/\[debateId\]/route.tsx
git commit -m "feat(live): add OG image generation API route for debate sharing"
```

---

### Task 4: Add Dynamic Metadata to Debate Viewer Route

**Files:**
- Modify: `aragora/live/src/app/(standalone)/debate/[[...id]]/page.tsx`

**Step 1: Update generateMetadata to fetch debate data and set dynamic OG tags**

Replace the existing `generateMetadata` function (lines 14-46) with one that:
- Fetches debate data from the backend API
- Sets topic-specific title and description
- Points `openGraph.images` to the OG image route
- Sets `twitter.card` to `summary_large_image`

```tsx
export async function generateMetadata(
  props: { params: Promise<{ id?: string[] }> }
): Promise<Metadata> {
  const params = await props.params;
  const debateId = params.id?.[0];

  if (!debateId) {
    return {
      title: 'ARAGORA Debate Viewer',
      description: 'Watch AI agents debate and reach consensus in real-time',
    };
  }

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
  let topic = `Debate ${debateId.slice(0, 12)}`;
  let verdict = '';
  let confidence = 0;
  let participantCount = 0;

  try {
    const res = await fetch(`${apiBase}/api/v1/debates/${debateId}`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) {
      const data = await res.json();
      topic = data.topic || data.question || topic;
      verdict = data.verdict?.replace(/_/g, ' ') || '';
      confidence = Math.round((data.confidence || 0) * 100);
      participantCount = data.participants?.length || 0;
    }
  } catch {
    // Use defaults
  }

  const truncatedTopic = topic.length > 70 ? topic.slice(0, 67) + '...' : topic;
  const description = [
    verdict && `Verdict: ${verdict}`,
    confidence && `${confidence}% confidence`,
    participantCount && `${participantCount} AI analysts`,
  ].filter(Boolean).join(' | ') || `ARAGORA debate analysis`;

  const ogImageUrl = `/api/og/${debateId}`;

  return {
    title: `ARAGORA | ${truncatedTopic}`,
    description,
    openGraph: {
      title: truncatedTopic,
      description,
      type: 'website',
      siteName: 'ARAGORA // LIVE',
      images: [{ url: ogImageUrl, width: 1200, height: 630, alt: `ARAGORA verdict: ${truncatedTopic}` }],
    },
    twitter: {
      card: 'summary_large_image',
      title: `ARAGORA | ${truncatedTopic}`,
      description,
      images: [ogImageUrl],
    },
  };
}
```

**Step 2: Verify types**

Run: `cd aragora/live && npx tsc --noEmit 2>&1 | head -20`
Expected: no type errors

**Step 3: Commit**

```bash
git add aragora/live/src/app/\(standalone\)/debate/\[\[...id\]\]/page.tsx
git commit -m "feat(live): add dynamic OG metadata to debate viewer for social sharing"
```

---

### Task 5: Create ShareToolbar Component

**Files:**
- Create: `aragora/live/src/components/ShareToolbar.tsx`
- Modify: `aragora/live/src/components/DebateResultPreview.tsx` (add ShareToolbar)

**Step 1: Create the ShareToolbar component**

```tsx
'use client';

import { useState, useCallback } from 'react';

interface ShareToolbarProps {
  debateId: string;
  topic: string;
}

export function ShareToolbar({ debateId, topic }: ShareToolbarProps) {
  const [copied, setCopied] = useState(false);

  const shareUrl = typeof window !== 'undefined'
    ? `${window.location.origin}/debate/${debateId}`
    : `/debate/${debateId}`;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const input = document.createElement('input');
      input.value = shareUrl;
      document.body.appendChild(input);
      input.select();
      document.execCommand('copy');
      document.body.removeChild(input);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [shareUrl]);

  const handleDownloadImage = useCallback(async () => {
    try {
      const res = await fetch(`/api/og/${debateId}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aragora-debate-${debateId.slice(0, 8)}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      // Silent fail
    }
  }, [debateId]);

  const tweetText = encodeURIComponent(`"${topic.slice(0, 200)}" — Multi-AI verdict on @aragora_ai`);
  const twitterUrl = `https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${tweetText}`;
  const linkedinUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleCopy}
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)] hover:text-[var(--acid-green)] transition-colors"
        title="Copy share link"
      >
        {copied ? 'Copied!' : 'Copy link'}
      </button>
      <button
        onClick={handleDownloadImage}
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-cyan)] hover:text-[var(--acid-cyan)] transition-colors"
        title="Download share image"
      >
        Image
      </button>
      <a
        href={twitterUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-cyan)] hover:text-[var(--acid-cyan)] transition-colors"
        title="Share on X"
      >
        X
      </a>
      <a
        href={linkedinUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--purple)] hover:text-[var(--purple)] transition-colors"
        title="Share on LinkedIn"
      >
        LinkedIn
      </a>
    </div>
  );
}
```

**Step 2: Add ShareToolbar to DebateResultPreview**

In `DebateResultPreview.tsx`, import `ShareToolbar` and render it above the summary bar:
```tsx
import { ShareToolbar } from './ShareToolbar';
// ...
{/* Share toolbar */}
<div className="flex items-center justify-between mb-2">
  <span className="text-xs font-mono text-[var(--text-muted)]">Share this debate</span>
  <ShareToolbar debateId={result.id} topic={result.topic} />
</div>
```

**Step 3: Verify types**

Run: `cd aragora/live && npx tsc --noEmit 2>&1 | head -20`
Expected: no type errors

**Step 4: Commit**

```bash
git add aragora/live/src/components/ShareToolbar.tsx aragora/live/src/components/DebateResultPreview.tsx
git commit -m "feat(live): add share toolbar with copy link, image download, and social sharing"
```

---

### Task 6: Create OracleBackground Canvas Particle System

**Files:**
- Create: `aragora/live/src/components/OracleBackground.tsx`

**Step 1: Create the canvas-based particle system**

```tsx
'use client';

import { useRef, useEffect, useCallback } from 'react';

const MODE_PARTICLE_COLORS: Record<string, [number, number, number]> = {
  consult: [200, 100, 200],  // magenta
  divine: [96, 165, 250],    // electric blue
  commune: [74, 222, 128],   // emerald green
};

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  opacity: number;
  baseOpacity: number;
}

interface OracleBackgroundProps {
  mode: 'consult' | 'divine' | 'commune';
}

export function OracleBackground({ mode }: OracleBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animRef = useRef<number>(0);
  const currentColorRef = useRef(MODE_PARTICLE_COLORS.consult);
  const targetColorRef = useRef(MODE_PARTICLE_COLORS.consult);
  const colorTransitionRef = useRef(1); // 0 = transitioning, 1 = done

  // Initialize particles
  const initParticles = useCallback((width: number, height: number) => {
    const count = 50;
    const particles: Particle[] = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: 1 + Math.random() * 2,
        opacity: 0.15 + Math.random() * 0.25,
        baseOpacity: 0.15 + Math.random() * 0.25,
      });
    }
    particlesRef.current = particles;
  }, []);

  // Update target color on mode change
  useEffect(() => {
    targetColorRef.current = MODE_PARTICLE_COLORS[mode] || MODE_PARTICLE_COLORS.consult;
    colorTransitionRef.current = 0;
  }, [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Check prefers-reduced-motion
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motionQuery.matches) {
      // Static gradient fallback
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        const [r, g, b] = MODE_PARTICLE_COLORS[mode] || MODE_PARTICLE_COLORS.consult;
        const grad = ctx.createRadialGradient(
          canvas.width / 2, canvas.height / 3, 0,
          canvas.width / 2, canvas.height / 3, canvas.width * 0.6
        );
        grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.04)`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      if (particlesRef.current.length === 0) {
        initParticles(canvas.width, canvas.height);
      }
    };
    resize();
    window.addEventListener('resize', resize);

    const animate = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      // Interpolate color
      if (colorTransitionRef.current < 1) {
        colorTransitionRef.current = Math.min(1, colorTransitionRef.current + 0.008);
        const t = colorTransitionRef.current;
        const [cr, cg, cb] = currentColorRef.current;
        const [tr, tg, tb] = targetColorRef.current;
        currentColorRef.current = [
          cr + (tr - cr) * t,
          cg + (tg - cg) * t,
          cb + (tb - cb) * t,
        ];
      }

      const [r, g, b] = currentColorRef.current;

      for (const p of particlesRef.current) {
        // Brownian drift
        p.vx += (Math.random() - 0.5) * 0.02;
        p.vy += (Math.random() - 0.5) * 0.02;
        // Damping
        p.vx *= 0.99;
        p.vy *= 0.99;

        p.x += p.vx;
        p.y += p.vy;

        // Wrap edges
        if (p.x < -10) p.x = width + 10;
        if (p.x > width + 10) p.x = -10;
        if (p.y < -10) p.y = height + 10;
        if (p.y > height + 10) p.y = -10;

        // Breathing opacity
        p.opacity = p.baseOpacity + Math.sin(Date.now() * 0.001 + p.x * 0.01) * 0.08;

        // Draw particle with radial gradient halo
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 6);
        grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${p.opacity})`);
        grad.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, ${p.opacity * 0.3})`);
        grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius * 6, 0, Math.PI * 2);
        ctx.fill();

        // Bright core
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.opacity * 1.5})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [mode, initParticles]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  );
}
```

**Step 2: Verify types**

Run: `cd aragora/live && npx tsc --noEmit src/components/OracleBackground.tsx 2>&1 | head -20`
Expected: no type errors

**Step 3: Commit**

```bash
git add aragora/live/src/components/OracleBackground.tsx
git commit -m "feat(live): add canvas-based bioluminescent particle system for Oracle background"
```

---

### Task 7: Refine Oracle Visual Layer

**Files:**
- Modify: `aragora/live/src/components/Oracle.tsx`

This is the largest task. Changes are organized by section within Oracle.tsx.

**Step 1: Import OracleBackground and add it to the render**

At the top of Oracle.tsx, add:
```tsx
import { OracleBackground } from './OracleBackground';
```

In the render section (around line 1396), replace the background atmosphere divs:
```tsx
{/* Background atmosphere — canvas particles */}
<OracleBackground mode={mode} />
```

Remove the old static background divs (the `oracle-bg` div and the scanline breathe div, lines 1396-1403).

**Step 2: Reduce BackgroundTentacles from 15 to 5 and improve them**

Replace the tentacle section (line 1406-1410):
```tsx
{/* Background tentacles — refined, fewer */}
<div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
  {Array.from({ length: 5 }, (_, i) => (
    <BackgroundTentacle key={i} index={i} />
  ))}
</div>
```

Update the `BackgroundTentacle` component (lines 79-101) to use gradient and glow:
```tsx
function BackgroundTentacle({ index }: { index: number }) {
  const left = 10 + (index * 20); // Evenly spaced
  const height = 200 + (index * 50) % 250;
  const duration = 8 + (index * 1.5) % 6;
  const delay = (index * 1.2) % 4;

  return (
    <svg
      className="absolute pointer-events-none"
      style={{
        left: `${left}%`,
        bottom: '-20px',
        width: '40px',
        height: `${height}px`,
        transformOrigin: 'bottom center',
        animation: `bg-tentacle-sway ${duration}s ease-in-out ${delay}s infinite`,
      }}
      viewBox={`0 0 40 ${height}`}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`tentacle-grad-${index}`} x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stopColor="var(--acid-green)" stopOpacity="0.2" />
          <stop offset="100%" stopColor="var(--acid-green)" stopOpacity="0" />
        </linearGradient>
        <filter id={`tentacle-glow-${index}`}>
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>
      <path
        d={`M20 ${height} C20 ${height * 0.7}, ${15 + index * 2} ${height * 0.4}, 20 0`}
        stroke={`url(#tentacle-grad-${index})`}
        strokeWidth="2"
        fill="none"
        filter={`url(#tentacle-glow-${index})`}
      />
      <path
        d={`M20 ${height} C20 ${height * 0.7}, ${15 + index * 2} ${height * 0.4}, 20 0`}
        stroke={`url(#tentacle-grad-${index})`}
        strokeWidth="1"
        fill="none"
      />
    </svg>
  );
}
```

**Step 3: Reduce FloatingEyes from 9 to 3 and add mode-colored iris**

Replace the floating eyes section (lines 1412-1421) with just 3 eyes:
```tsx
{/* Floating eyes — refined, fewer, mode-colored */}
<FloatingEye delay={0} x={5} y={20} size={1.1} mode={mode} />
<FloatingEye delay={3} x={92} y={35} size={0.9} mode={mode} />
<FloatingEye delay={6} x={8} y={75} size={1.0} mode={mode} />
```

Update the `FloatingEye` component to accept `mode` and use mode-colored iris:
```tsx
const EYE_IRIS_COLORS: Record<string, string> = {
  consult: 'rgba(200, 100, 200, 0.6)',
  divine: 'rgba(96, 165, 250, 0.6)',
  commune: 'rgba(74, 222, 128, 0.6)',
};

function FloatingEye({ delay, x, y, size, mode }: { delay: number; x: number; y: number; size: number; mode?: string }) {
  const irisColor = EYE_IRIS_COLORS[mode || 'consult'] || EYE_IRIS_COLORS.consult;

  return (
    <div
      className="absolute pointer-events-none select-none"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        width: `${size * 10}px`,
        height: `${size * 10}px`,
        borderRadius: '50%',
        background: `radial-gradient(circle, ${irisColor} 0%, ${irisColor.replace('0.6', '0')} 70%)`,
        opacity: 0,
        animation: `eye-blink-bg 4s ease-in-out ${delay}s infinite`,
        boxShadow: `0 0 ${size * 15}px ${irisColor}`,
      }}
      aria-hidden="true"
    />
  );
}
```

**Step 4: Add response emergence effect and text luminescence**

In the CSS section (inside the `<style>` tag, around line 1334), add:
```css
.oracle-message-enter {
  animation: oracle-msg-enter 0.3s ease-out forwards;
}
@keyframes oracle-msg-enter {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
```

Replace `prophecy-reveal` on oracle message divs in the chat area with `oracle-message-enter` and add text-shadow glow. In the oracle message rendering (around line 1748):
```tsx
<div
  className="border-l-2 border-[var(--acid-magenta)] pl-4 py-3 pr-3 text-sm leading-relaxed whitespace-pre-wrap rounded-r-lg oracle-message-enter"
  style={{
    color: 'var(--text)',
    backgroundColor: 'var(--surface)',
    textShadow: `0 0 20px ${MODE_COLORS[msg.mode]?.glow || 'rgba(200,100,200,0.1)'}`,
  }}
>
```

Note: Also change the hardcoded `color: '#2d1b4e', backgroundColor: 'rgba(200, 235, 210, 0.9)'` on message bubbles to use theme-aware colors instead. This affects tentacle messages (line 194), debate event messages (line 347), and oracle messages (line 1751). Replace with:
```tsx
style={{ color: 'var(--text)', backgroundColor: 'var(--surface)' }}
```

**Step 5: Update mode button active state with animated glow**

In the `ModeButton` component, update the active button style to pulse:
```tsx
style={{
  borderColor: active ? c.border : 'rgba(255,255,255,0.1)',
  boxShadow: active ? `0 0 12px ${c.glow}` : 'none',
  animation: active ? 'mode-btn-pulse 2s ease-in-out infinite' : 'none',
}}
```

Add to the `<style>` tag:
```css
@keyframes mode-btn-pulse {
  0%, 100% { box-shadow: 0 0 12px var(--pulse-color); }
  50% { box-shadow: 0 0 20px var(--pulse-color); }
}
```

**Step 6: Update input placeholder per mode and add glow**

The textarea already has mode-specific placeholders (line 1632-1637). Update the textarea styling to include a mode-colored glow:
```tsx
style={{
  boxShadow: `0 0 15px ${MODE_COLORS[mode]?.glow || 'rgba(200,100,200,0.08)'}`,
}}
```

**Step 7: Verify types**

Run: `cd aragora/live && npx tsc --noEmit 2>&1 | head -20`
Expected: no type errors

**Step 8: Commit**

```bash
git add aragora/live/src/components/Oracle.tsx
git commit -m "feat(live): refine Oracle visual layer with particles, smoother tentacles, and luminescent text"
```

---

### Task 8: Verify Everything Builds

**Files:** none (verification only)

**Step 1: Run TypeScript check**

Run: `cd aragora/live && npx tsc --noEmit 2>&1 | tail -5`
Expected: no errors (or only pre-existing unrelated errors)

**Step 2: Run Next.js build**

Run: `cd aragora/live && npx next build 2>&1 | tail -20`
Expected: build succeeds

**Step 3: Verify all new files exist**

Run:
```bash
ls -la aragora/live/src/components/PerspectiveCard.tsx \
       aragora/live/src/components/ShareToolbar.tsx \
       aragora/live/src/components/OracleBackground.tsx \
       aragora/live/src/app/api/og/*/route.tsx
```
Expected: all 4 files exist

**Step 4: Commit any final fixes**

If build revealed issues, fix and commit.
