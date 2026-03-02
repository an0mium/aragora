# Landing Page Tri-Theme Redesign

**Date:** 2026-03-01
**Status:** Approved
**Goal:** Redesign aragora.ai landing page with three selectable visual themes sharing one lean 5-section layout

---

## Theme Architecture

Three themes selectable via a segmented control in the header:

| Theme | Default For | Font | Palette | Personality |
|-------|-------------|------|---------|-------------|
| Warm | Light mode (default) | Inter | Cream + forest green | Approachable, Notion-like |
| Dark | Dark mode | JetBrains Mono | Near-black + acid green | Terminal hacker, GitHub dark |
| Professional | Third option | Inter | White + slate + emerald | Enterprise, Stripe-like |

### Theme Selector

3-way segmented control in header, right side. Replaces current sun/moon toggle.
Persists to localStorage. Icons: sun (warm), moon (dark), diamond (professional).

---

## Color Systems

### Warm (default)
```
--bg:           #faf9f7
--surface:      #ffffff
--border:       #e8e4de
--text:         #2c2a28
--text-muted:   #736f68
--accent:       #1e7033
--accent-hover: #166028
--link:         #166a7a
```

### Dark (refined hacker)
```
--bg:           #0a0a0a
--surface:      #111111
--border:       #1a1a1a
--text:         #f0f0f0
--text-muted:   #a0a0a0
--accent:       #39ff14
--accent-hover: #2dd911
--link:         #00cccc
```

### Professional
```
--bg:           #ffffff
--surface:      #f8fafc
--border:       #e2e8f0
--text:         #0f172a
--text-muted:   #64748b
--accent:       #059669
--accent-hover: #047857
--link:         #2563eb
```

---

## Typography Per Theme

| Element | Warm | Dark | Professional |
|---------|------|------|-------------|
| Font family | Inter | JetBrains Mono | Inter |
| Headline | 48px semibold | 42px bold | 48px semibold |
| Subtitle | 18px regular | 16px regular | 18px regular |
| Body | 16px | 14px | 16px |
| Section labels | 12px uppercase tracking-wide | 11px uppercase | 12px uppercase |

---

## Theme-Specific Visual Elements

### Dark theme ONLY (keeps terminal DNA):
- ASCII art ARAGORA banner in hero
- CRT scanline overlay (subtle)
- Glow effects on accent text and primary CTA
- Terminal chrome: `>` prompts, `[bracketed]` labels, `[01]` step numbers
- 0px border-radius (sharp corners everywhere)
- No shadows (borders only)

### Warm theme ONLY:
- Clean "Aragora" wordmark (no ASCII art)
- Rounded-xl corners (12-16px)
- Warm soft shadows on cards
- No terminal chrome
- No glow/CRT effects

### Professional theme ONLY:
- Clean "Aragora" wordmark (no ASCII art)
- Rounded-lg corners (8px)
- Subtle cool shadows
- No terminal chrome
- No glow/CRT effects

---

## Page Sections (5 lean sections, shared structure)

### Section 1: Hero (with debate input)

Content:
- Headline: "Don't trust one AI." / "Make them argue." (second line accent-colored)
- Subtitle: "Multiple AI models debate your question, stress-test each answer, and deliver an audit-ready verdict you can defend."
- Debate input form (ported from old LandingPage.tsx with backend connection)
- Primary CTA: "Run a free debate"
- Example topic links (3 clickable examples)

Theme variations:
- Warm: rounded-xl input, soft shadow, solid green button rounded-full
- Dark: ASCII banner above headline, sharp input, 1px border, glow green button, `>` prompt prefix
- Pro: rounded-lg input, subtle shadow, solid emerald button rounded-lg

### Section 2: Problem (why single AI fails)

Content:
- Section label: "THE PROBLEM"
- Statement: "A single AI hallucinates, agrees with you, and contradicts itself. Adversarial debate fixes all three."
- 3 cards: Hallucination, Sycophancy, Inconsistency
  - Each: title, 2-line description

Theme variations:
- Warm: white cards, rounded-xl, warm shadow, green left-border or top-border
- Dark: #111 cards, sharp, 1px border, thin green top-accent line
- Pro: slate-50 cards, rounded-lg, slate border, emerald top-accent

### Section 3: How It Works (3 steps)

Content:
- Section label: "HOW IT WORKS"
- Step 01: "You ask a question" + description
- Step 02: "AI agents debate it" + description (mention Claude, GPT, Gemini, Mistral)
- Step 03: "You get a decision receipt" + description (mention evidence chains, confidence scores, dissent)

Theme variations:
- Warm: numbers in forest green, body in warm text
- Dark: numbers in acid green mono `[01]`, body in muted gray, `>` prefix on titles
- Pro: numbers in emerald, body in slate text

### Section 4: Pricing

Content:
- 3 tiers: Free ($0) / Pro ($49/seat/mo, highlighted) / Enterprise (Custom)
- Free: 10 debates/month, 3 agents, Markdown receipts, demo mode
- Pro: Unlimited, 10 agents, all export formats, CI/CD, channels, memory
- Enterprise: SSO/RBAC/encryption, compliance frameworks, self-hosted
- "Bring your own API keys" note

Theme variations:
- Warm: white cards, green border on Pro, rounded
- Dark: dark cards, green border on Pro, sharp, glow on Pro header
- Pro: white/slate cards, emerald border on Pro, rounded-lg

### Section 5: Footer

Content:
- "No signup required. First result in under 30 seconds."
- CTAs: [Try it now] [Create an account]
- Nav links: About, Pricing, Docs, Support
- Tagline: "AI decisions you can trust."

---

## Technical Implementation

### Files to modify/create:
1. `src/context/ThemeContext.tsx` — extend to 3 themes
2. `src/app/globals.css` — add CSS variables for all 3 themes
3. `src/components/landing/ThemeSelector.tsx` — new 3-way selector
4. `src/components/landing/LandingPage.tsx` — rewrite with 5 sections
5. `src/components/landing/HeroSection.tsx` — rewrite with debate input
6. `src/components/landing/ProblemSection.tsx` — new (replaces WhyAragora)
7. `src/components/landing/HowItWorksSection.tsx` — new (replaces DebateProtocol)
8. `src/components/landing/PricingSection.tsx` — rewrite
9. `src/components/landing/Footer.tsx` — simplify
10. `src/components/landing/Header.tsx` — simplify nav + add ThemeSelector
11. `src/app/(standalone)/landing/page.tsx` — wire to new modular landing

### Files to remove/deprecate:
- `src/components/landing/VerticalCards.tsx` (cut)
- `src/components/landing/CapabilitiesSection.tsx` (cut)
- `src/components/landing/TrustSection.tsx` (cut)
- `src/components/landing/SectionHeader.tsx` (inlined)

### Font loading:
- Inter: loaded via next/font/google for Warm and Professional
- JetBrains Mono: already loaded, used for Dark theme

### Theme persistence:
- localStorage key: `aragora-theme`
- Values: `'warm' | 'dark' | 'professional'`
- Default: `'warm'`
- System preference: if user hasn't chosen, map `prefers-color-scheme: dark` to Dark theme
