# Landing Page Debate UX Redesign

## Goal

Fix the landing page debate experience so that: visitors never see a timeout error, ambiguous questions get clarified before debate, results lead with a synthesized answer instead of raw agent dumps, and real streaming progress replaces fake phases.

## Context

The current landing page submits raw user text directly to the playground debate endpoint with no interpretation step, no client-side timeout, fake progress phases on fixed timers, and a verbose result view that dumps all proposals expanded. This produces three failure modes observed in production:

1. **Timeouts** — "Operation timed out" when backend live debate exceeds 15s or mock fallback fails
2. **Misinterpretation** — "should I cook my chickens" becomes a debate about live animal cruelty instead of reheating nuggets
3. **Verbose results** — Three walls of agent prose before the user finds the actual answer

## Principles

- **Use real intelligence, not regex** — Frontier LLM models for all classification, disambiguation, and synthesis. No heuristic shortcuts.
- **Never show errors on the landing page** — Always produce a result, even if degraded/mock.
- **Honest progress** — Show what's actually happening, not fake phases.

---

## Section 1: Timeout & Reliability

### Client-Side Timeout
- Add `AbortController` with **180s (3 minute)** timeout to the fetch in `HeroSection.tsx`
- On timeout, show friendly message + retry button (not raw error text)

### Backend Live Debate Timeout
- Increase `_LIVE_TIMEOUT` in `playground.py` from 15s to **90s**
- Allows larger text inputs and multi-round debates to complete without premature cutoff

### Fallback Hardening
- Ensure mock fallback in `playground.py._run_debate()` catches ALL exception types (TimeoutError, ConnectionError, OSError, any Exception)
- If mock also fails, return a pre-baked sample result with `mock_fallback=true` and `mock_fallback_reason="service_unavailable"`
- The landing page should NEVER return an error response — every code path produces a result

### Files Modified
- `aragora/live/src/components/landing/HeroSection.tsx` — client timeout
- `aragora/server/handlers/playground.py` — backend timeout + fallback hardening

---

## Section 2: Ambiguity Detection (Pre-Debate Intelligence Gate)

### New Endpoint
- `POST /api/v1/playground/assess`
- Takes `{ question: string }`
- Calls a frontier model (Claude Sonnet or fastest available) with a focused prompt asking whether the question is clear or ambiguous
- Returns `{ clear: boolean, interpretations?: string[], suggested_topic?: string }`
- **Timeout**: 5s. On failure, returns `{ clear: true }` (skip disambiguation, never block)

### Frontend Flow
1. User submits question on landing page
2. Call `/assess` (1-3s)
3. If `clear: true` → proceed directly to debate with zero friction
4. If `clear: false` → show inline: "This could mean a few things:" with 2-3 clickable interpretation options
5. User picks one → that becomes the debate topic
6. If assess call fails → skip it, debate the raw question

### Model Selection
- Use the fastest frontier model available via the existing agent fallback chain (Anthropic → OpenRouter)
- This is a classification task — speed > depth

### Files Created
- `aragora/server/handlers/playground_assess.py` — new handler
- Assessment prompt template (inline in handler or in `aragora/prompt_engine/`)

### Files Modified
- `aragora/live/src/components/landing/HeroSection.tsx` — pre-debate assess call + disambiguation UI
- `aragora/server/unified_server.py` — register new route

---

## Section 3: Real Streaming Progress

### WebSocket Integration
- Wire up the existing spectate WebSocket (already fires `debate_start`, `agent_message`, `critique`, `vote`, `consensus`, `debate_end` events) to the landing page progress UI

### Progress States (driven by real backend events)
1. **"Assessing question..."** — during `/assess` call
2. **"Asking agents..."** — `debate_start` received, show agent names
3. **"[Agent] is responding..."** — `agent_message` arrives, pulse that agent's colored chip
4. **"Round N: Critiques..."** — `critique` events arrive
5. **"Building consensus..."** — `vote`/`consensus` events arrive
6. **"Done"** — `debate_end`, transition to result card

### Multi-Round Display
- Show "Round 1 of N" → "Round 2 of N" as round events arrive

### Elapsed Timer
- Real clock from submission time. No fake "~15s remaining" countdown.

### Fallback
- If WebSocket connection fails, show simple spinner + elapsed time (honest, not fake phases)
- Debate still completes via HTTP POST — streaming is purely for progress display

### Files Modified
- `aragora/live/src/components/landing/HeroSection.tsx` — replace fake phase UI with real event-driven progress
- May need a lightweight hook (e.g., `useLandingDebateProgress.ts`) to manage WebSocket subscription scoped to a single debate

---

## Section 4: Compact Inline Result Card

### New Component: `CompactDebateResult`
Used on the landing page only. The existing `DebateResultPreview` stays for the full `/debate/{id}` page.

### Layout (top to bottom)
1. **Interpretation line** (conditional): "Aragora interpreted this as: [topic]" — small, muted text. Only shown if disambiguation was used.
2. **TL;DR answer card**: Green-bordered rounded card with one-sentence synthesized answer (see Section 5). NOT the raw `final_answer`.
3. **Metadata row**: `{confidence}% confidence · {agent_count} agents · {rounds} round(s) · {duration}s`
4. **Agent chips**: Colorful rounded pills per agent. **Clickable** — clicking expands a collapsible card below showing that agent's proposal (first ~3 lines, "Read more" to expand fully). For multi-round debates, show which round the proposal is from.
5. **Receipt row**: Receipt hash + timestamp, muted text
6. **Actions**: "View full debate →" link to `/debate/{id}` + "Share" button

### What's NOT in the compact view
- Critiques
- Votes
- Dissenting views
- Full verdict section (the TL;DR replaces it)

These all live on the full debate page only.

### Files Created
- `aragora/live/src/components/landing/CompactDebateResult.tsx`

---

## Section 5: TL;DR Synthesis

### Post-Debate Synthesis
- After debate completes in `playground.py`, make one additional frontier model call
- Prompt: "Given these agent proposals and the original question, write a single-sentence direct answer. Be practical, not philosophical."
- Input: all proposals + original question
- Output: `tldr` field added to response dict

### Model & Performance
- Same fast frontier model as ambiguity detection
- **Timeout**: 5s. On failure, fall back to truncating `final_answer` to first sentence.
- **Cost**: ~$0.002 per synthesis (negligible vs ~$0.04 debate cost)

### Storage
- `tldr` persisted alongside debate result so `/debate/{id}` page can also use it as a header

### Files Modified
- `aragora/server/handlers/playground.py` — add synthesis step after `_run_debate()`, before `_persist_and_respond()`

---

## Section 6: Full Debate Page Improvements

### Modifications to existing page
- **Add TL;DR card at top** — same green card from compact view, so full page also leads with the answer
- **Add interpretation line** — if disambiguation was used, show it here too
- **Proposals expanded by default** (opposite of compact view behavior)
- **Wider layout** — increase max-width from ~800px to 960px for better readability of markdown content

### No new component needed
- Modifications to existing `/debate/[[...id]]` route and `DebateResultPreview` component

### Files Modified
- `aragora/live/src/components/DebateResultPreview.tsx` — add TL;DR card, interpretation line, wider layout
- `aragora/live/src/app/(standalone)/debate/[[...id]]/` — layout width adjustment

---

## Implementation Order

1. **Timeout & reliability** (Section 1) — highest impact, simplest change
2. **TL;DR synthesis** (Section 5) — needed before the compact card can work
3. **Compact result card** (Section 4) — the main UX improvement
4. **Ambiguity detection** (Section 2) — new endpoint + frontend flow
5. **Real streaming progress** (Section 3) — WebSocket integration
6. **Full debate page improvements** (Section 6) — polish

---

## Out of Scope

- Changing the debate engine itself (agent selection, consensus algorithm)
- Backend interrogator integration beyond the landing page
- Mobile-specific layouts (follow-up work)
- Authentication/signup flow changes
