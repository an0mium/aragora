# aragora.ai — Stranger-Journey Audit 2026-05-19

**Date:** 2026-05-19
**Auditor:** claude-B061F80D, with a real browser via Playwright (no spoofing)
**Trigger:** founder reframe — "more important than the demo is an actual
working product on the aragora.ai website that can be used." The 30-day
strategic assessment (`docs/status/PROJECT_ASSESSMENT_2026-05-19_30D.md`)
optimized for thesis novelty and did not check whether the basic product
surface works for a stranger. This audit closes that gap.
**Scope:** anonymous unauthenticated visitor, on a clean browser, hitting
the live `aragora.ai` site. Authenticated app surface (`/(app)/*`) and the
docs subdomain (`docs.aragora.ai`) are out of scope for this pass.

---

## TL;DR

The site is **partly working**. The core debate machinery is real and
returns real results in under 30 seconds. The biggest single
stranger-blocker is **the homepage's most prominent CTA goes to a
"Debate not found" error page**. After that, the next-tier problems are
about polish, consistency, and the differentiated artifact (the
cryptographic receipt) being invisible in the surfaces that demo the
product.

**The product is usable enough that a stranger can have a real
experience with it. It is not yet polished enough that a stranger
would conclude the team has shipped something.**

### Findings by priority

| P | Count | Examples |
|---|---:|---|
| **P0 — stranger-blocking** | **1** | Homepage "Try a Demo Debate" CTA → "Debate not found" error |
| **P1 — silent breakage / missing differentiator** | **5** | Docs nav broken (CORS); receipt artifact invisible on /demo/ and /playground/; live-bridge promised but always empty; engineering jargon in /try subtitle; landing page polls /spectate/status repeatedly |
| **P2 — polish / consistency** | **3** | Three different navbars across routes; landing page "Start Debate" button disabled even with placeholder text; "Try a Demo Debate" duplicates "Run your own debate" |
| **P3 — missing surface** | **3** | No "verify the receipt hash yourself" UX; no public list of which models actually ran in a given debate (vendor confirmation); no obvious "what just happened?" walkthrough after a demo completes |

---

## Method

- Real Chromium browser via Playwright MCP
- Pages visited as anonymous user: `/`, `/landing/`, `/demo/`, `/playground/`,
  `/quickstart/`, `/try/`, `/signup/`, `/debate/LV-20260520-4dfda6/`
- For each page: screenshot, accessibility snapshot, console errors,
  network requests (filtered to relevant endpoints)
- One end-to-end interactive flow: homepage "Try a Demo Debate" button
- One backend completion flow: /demo/ POSTs `/api/v1/playground/debate`,
  result observed after ~20s wall-clock
- Time used: ~25 minutes of focused interactive audit

---

## P0 — Stranger-blocking

### P0.1 — Homepage's most prominent demo CTA renders "Debate not found"

**Where:** `https://aragora.ai/landing/` (the homepage). The most
prominent CTA below the headline is "Try a Demo Debate". Click it.

**What happens:**

1. Click takes you to `https://aragora.ai/debate/LV-20260520-4dfda6/`
2. Page renders `> ERROR — Debate not found` with a "[RETURN HOME]" link
3. Browser console:
   ```
   WebSocket connection to 'wss://api.aragora.ai/ws/spectate?debate_id=LV-20260520-4dfda6'
   failed: Error during WebSocket handshake: Unexpected response code: 400
   ```

**Why it's P0:** the headline button on the home page, labeled in a way
that any visitor will read as "let me see this work," takes the visitor
to a broken page within 1 second. Whatever else is true about the site,
a stranger's first interaction proves it doesn't work.

**Diagnosis hypothesis:** the click handler appears to generate a fresh
debate ID client-side (`LV-YYYYMMDD-XXXXXX`) and immediately navigate to
`/debate/<id>/`, where the page tries to open a WebSocket to fetch the
debate state — but no debate with that ID actually exists yet because
nothing on the backend created it. Either the click handler is supposed
to POST to create the debate first (and the POST isn't firing), or the
`/debate/<id>/` page is supposed to handle a "pre-created, not-yet-running"
state with a loading UI (and that branch isn't implemented).

**Smallest fix:** wire the "Try a Demo Debate" button to navigate to
`/demo/` (the canonical-question runner that already works end-to-end)
instead of creating a fresh broken debate ID. That's a one-line href
change. The "I want my own question" intent is already served by the
"Ask Your Own Question" button further down — and by `/try/` and
`/playground/`.

**Better fix:** the `/debate/<id>/` "Debate not found" branch should
detect a fresh-but-empty debate and either redirect to a creation flow
or show a "still warming up, the demo will land here in ~20s"
intermediate state instead of an error.

---

## P1 — Silent breakage / missing differentiator

### P1.1 — `/docs/` nav link is broken on every page (CORS)

**Where:** every page that includes the "Docs" nav link (landing,
playground, others).

**What happens:** clicking "Docs" triggers a request to
`aragora.ai/docs/?_rsc=…` which is redirected to
`docs.aragora.ai/?_rsc=…`. The Next.js RSC fetch to the subdomain fails
CORS preflight; console shows:

```
Access to fetch at 'https://docs.aragora.ai/?_rsc=…' (redirected from
'https://aragora.ai/docs/?_rsc=…') from origin 'https://aragora.ai' has
been blocked by CORS policy: Response to preflight request doesn't pass
access control check.
Failed to load resource: net::ERR_FAILED @ https://docs.aragora.ai/?_rsc=…
```

The user-visible result is that the docs page may eventually render
after the failed RSC fetch, but the browser console shows real errors on
every page load, every time the docs link is hovered or clicked.

**Smallest fix:** rewrite or proxy `/docs/` so it stays same-origin
(serve docs from `aragora.ai/docs/*` via reverse proxy rather than
cross-origin redirect to `docs.aragora.ai`). Alternatively, configure
CORS on `docs.aragora.ai` to allow the `aragora.ai` origin (but the
RSC-fetch shape is sensitive enough that proxying is safer).

### P1.2 — The differentiated artifact (the receipt) is invisible on `/demo/` and `/playground/`

**Where:** `/demo/` after the live debate completes. The result panel
shows recommendation, why/main caution, full verdict, multi-agent
positions (GPT, Grok 3, Claude), convergence confidence, "View Sharable
Result" — but no receipt-shaped artifact. No receipt_id, no
artifact_hash, no signature_algorithm, no "verify the hash" affordance.

**Why it's P1, not P0:** the surface works (debate runs, result lands).
But the marketing position is "audit-ready decision receipt with
evidence chains, confidence scores, and dissenting views preserved."
The visitor sees the result but doesn't see the receipt. The
differentiation pitch and the visible surface don't match.

**Smallest fix:** PR #7386 (just opened, `claude/sample-receipt-public-page-20260520`)
adds a `/sample-receipt/` page that renders a real receipt artifact with
a field guide and a "verify the hash yourself" snippet. Linking to that
from the /demo/ completion screen would close the loop. (#7386 is itself
the stop-gap, not the structural fix — the structural fix is rendering
the actual receipt for the just-completed debate inline.)

**Better fix:** the /demo/ completion screen should include a collapsed
"Receipt" panel showing `receipt_id`, `artifact_hash`,
`signature_algorithm`, and a "verify" link, expandable to show the full
JSON.

### P1.3 — Live-bridge panel promises live debates but is always empty

**Where:** `/landing/`, the "LIVE DEBATE / FOLLOWING LIVE BRIDGE" panel
and the "SEE IT IN ACTION" section below.

**What happens:** both panels render copy like *"The bridge is online.
This panel will attach as soon as a public debate starts emitting
events"* and *"Bridge ready / Public spectate is online, but no recent
live debate activity is visible yet"*. The page falls back to a
hard-coded looping sample exchange ("Should a fast-growing software org
split the monolith now or sequence the migration later?") with canned
agent text.

**Why it's P1:** the page tells the visitor "watch live debates here"
and there are never any to watch. The fallback is well-designed but
the framing makes the absence visible. Visitor read: *"this is supposed
to be a busy product but nothing is happening."*

**Smallest fix:** either (a) actually emit at least one looping public
demo into the spectate stream so the live panel has something to show,
or (b) collapse the live-bridge framing to just the fallback panel
labeled as "Sample debate" without the "waiting for live activity"
wrapper.

### P1.4 — `/try/` subtitle exposes internal jargon to public users

**Where:** `https://aragora.ai/try/`. Subtitle:

> "Ask your own question through the public beta flow. `/try` keeps
> rate limiting, persistence, and replay/share behavior intact while
> `/demo` stays focused on the canonical live proof."

**Why it's P1:** this reads as a release-notes commit message left in
the user-facing copy. A stranger doesn't know what "the canonical live
proof" is; doesn't care that `/try` keeps rate limiting "intact";
doesn't have a model of why there are two routes. Visitor read:
*"this product is not ready for me."*

**Smallest fix:** replace with one stranger-oriented sentence: *"Ask
any decision question and watch 3 AI models debate it. No signup
required."* The internal distinction between /try and /demo can be
captured in a code comment or an internal doc.

### P1.5 — Landing page polls `/api/v1/spectate/status` repeatedly

**Where:** `/landing/`. Observed ~8+ consecutive `/api/v1/spectate/status`
GET requests within ~20 seconds of page load, all 200.

**Why it's P1:** this is server-cost waste and a tiny bit of
client-visible "this thing is busy" smell. For an SaaS that bills on
provider API costs but pays for its own infra, polling status every
~2s on every landing-page visitor adds up. It's also visible in any
network panel a curious visitor opens.

**Smallest fix:** WebSocket or SSE for the live-bridge subscription
rather than polling. If polling is intentional, increase the interval
to ~15s and only poll while the page is foregrounded
(`document.visibilityState === 'visible'`).

---

## P2 — Polish / consistency

### P2.1 — Three different navbars across public routes

| Route | Nav contents |
|---|---|
| `/landing/` | ARAGORA / How it works / Quickstart / Docs / Pricing / Log in + theme picker |
| `/playground/` | DEBATE DEMO / REST API / WEBSOCKET / API DOCS / SIGN UP FREE |
| `/demo/`, `/try/`, `/quickstart/` | ARAGORA // LIVE + Warm/Dark/Pro theme + (/try beta tag on demo) / Get started free |
| `/signup/` | ARAGORA + "Already have an account? LOG IN" only |

A stranger navigating across these pages experiences a different shell
each time, often without the same nav links. Inconsistency reads as
"this is multiple half-finished products glued together." Particularly
jarring: `/playground/` has API/WEBSOCKET nav links presumably useful
to developers but invisible from `/landing/`.

**Smallest fix:** pick one global nav and apply it. If `/playground/`
really needs developer-specific links (REST API, WEBSOCKET, API DOCS),
those should live in a sub-nav within the page, not replace the global
nav.

### P2.2 — Landing-page "Start Debate" button is disabled even with placeholder text

**Where:** `/landing/` hero. The textbox has placeholder text *"Is it
worth switching from React to Svelte for our dashboard?"* and the
"Start Debate" button next to it is **disabled** until the visitor
types something. The visual affordance suggests the placeholder *is*
the question, so a visitor might click expecting it to use that.

**Smallest fix:** either (a) make the button active with the
placeholder treated as a default question, or (b) make the disabled
state clearer with helper text ("Type a question, then Start Debate")
and gray out the placeholder more strongly so it doesn't look like a
filled value.

### P2.3 — Redundant CTAs lead to the same flow with slightly different framings

The landing page has three primary CTAs that all lead to
debate-flow-adjacent things: "Start Debate" (disabled), "Try a Demo
Debate" (broken — see P0.1), and "Run your own debate" further down.
The "SEE IT IN ACTION" section also has "Open full spectate view" and
"Run your own debate" together. Eight unique routes are linked from
the landing page alone (`/landing/`, `/spectate/`, `/demo/`,
`/quickstart/`, `/docs/`, `/pricing/`, `/playground/`, `/signup/`,
`/login/`, `/about`).

A visitor can't tell from naming alone which CTA leads to what. /demo/
and /playground/ and /try/ are all visually distinct but functionally
overlapping.

**Smallest fix:** decide which route is the canonical "stranger first
clicks here" surface, label every other entry point as obviously
secondary, and probably collapse /try/ + /demo/ + /playground/ into one
route with internal modes.

---

## P3 — Missing surface (for the differentiated thesis to land)

### P3.1 — No "verify the receipt hash yourself" UX anywhere on the site

**Where:** missing. The thesis claim is "cryptographic decision
receipts you can verify offline." Nowhere on the public site is a
visitor invited or shown how to verify a hash.

**Smallest fix:** PR #7386 includes a "verify the hash yourself" code
snippet on the proposed `/sample-receipt/` page. Once merged, that's
the first real instance.

### P3.2 — No vendor-confirmation surface

**Where:** the /demo/ result panel shows agent positions labeled "GPT
/ Grok 3 / Claude" but doesn't tell the visitor which model versions
those are (gpt-5? Claude 4.7? Grok 3 from 2026-04?) or that they're
served by different vendors. Yet vendor-diversity is the entire
adversarial-debate value proposition.

**Smallest fix:** on the result panel, expand each agent label to
include the model identifier and the vendor (e.g., "anthropic /
claude-opus-4.7-1m" or "openai / gpt-5"). Cheap, ~15 lines of UI.

### P3.3 — No "what just happened?" walkthrough after a demo completes

**Where:** /demo/ completion. The visitor sees the result but isn't
walked through "here's the recommendation, here's why each agent
disagreed, here's the dissent, here's the receipt." It's a lot of info
to absorb at once and most of the value proposition is in the
disagreement-being-recorded, which the panel doesn't draw attention to.

**Smallest fix:** an optional collapsible "How to read this result"
section explaining what the visitor is looking at, why the disagreement
matters, and what the receipt is for. Could double as the entry point
to the receipt sample.

---

## What works (for balance — these are honest positives)

- **The backend actually runs real debates.** `/demo/` POSTs to
  `/api/v1/playground/debate`, gets a 200 back in ~10-20s, and
  displays a real multi-agent result with distinct agent positions,
  recommendation, dissent, and confidence. The hardest, most
  product-defining part of the system is real.
- **`/api/health` returns 200.** Production API is up.
- **`/signup/` is polished.** Google / GitHub / Microsoft OAuth +
  email/password. Standard SaaS shape, no obvious breakage. (Did not
  test actual signup completion.)
- **`/playground/` is well-designed.** Clean entry, 5 canned example
  questions to pick from, a "type your own" affordance, a big START
  DEMO button. Best stranger-entry surface on the site.
- **The fallback sample debate is well-written.** The landing page's
  looping fallback debate on monolith-vs-microservices reads as
  thoughtful, not template. If the live bridge were re-framed (P1.3),
  this fallback is good enough to be the actual sample.
- **The site is fast.** Page loads sub-second; the /demo/ completion is
  bottlenecked by the actual provider calls, not by client-side
  performance.

---

## Recommendation: the smallest set of changes to make the stranger
journey actually work

**If only one change ships:** fix P0.1 (rewire "Try a Demo Debate" to
`/demo/`). A one-line change closes the worst single bug.

**Two-day polish pass that closes the credibility gap:**

1. P0.1 — rewire homepage CTA (1 line, 5 min)
2. P1.3 — collapse the empty live-bridge into a sample-only panel
   (drop the "waiting for live" copy, keep the fallback) — 30 min of
   copy + UI work
3. P1.4 — replace the /try subtitle with a stranger-readable one
   (1 sentence, 5 min)
4. P2.1 — unify the nav across landing / playground / demo / try /
   quickstart — 1-2 hours
5. P1.2 — show the receipt on /demo/ completion (either inline panel
   or link to the new `/sample-receipt/` from #7386) — 1-2 hours
6. P1.1 — fix the docs CORS by proxying same-origin — 1-2 hours
   depending on deployment

That's a single ~6-hour session that takes the site from "broken on
first click" to "polished enough that a stranger would believe the
team shipped something." None of it requires new product work — every
fix is to existing surfaces.

**Out of scope for this audit:** the bigger product questions named in
the founder's reframe — Nomic-loop self-improvement end-to-end,
production infrastructure reliability under load, 60-day roadmap
maximalist vision. Those are own-track investigations. This audit
narrowly answers "does the existing public site actually work for a
stranger today?" — and the answer is "mostly yes, with one P0 and
several P1s blocking polished credibility."

---

## What this audit does not cover

- Authenticated app surface (`/(app)/*`) — visitor cannot reach it
  without account
- Docs subdomain (`docs.aragora.ai`) — separate Docusaurus deployment,
  needs its own audit
- Spectate flow with an actual live debate — none in flight during
  audit
- Mobile responsiveness — desktop browser only
- Performance under load — single-user session
- Signup completion (didn't create an account)
- Pricing page deep-read
- Login flow (no credentials)
- Slow/flaky-network behavior
- The "Connect OpenRouter for instant setup" button (didn't click)
- Browser-back behavior, deep linking, share URLs

Each of these would be its own next-tier audit. Listed here so they
don't get silently forgotten.

---

## Provenance

- Audit screenshots: `demo-page.png`, `demo-after-20s.png`,
  `playground.png`, `signup.png`, `try-page.png` (local artifacts,
  not committed)
- Browser console / network logs: in
  `.playwright-mcp/console-*.log` (local, not committed)
- Pair PR: `#7386` (`/sample-receipt/` page) — partial fix for P1.2
  and P3.1
- Parent assessment: `docs/status/PROJECT_ASSESSMENT_2026-05-19_30D.md`
  (#7373, merged)
- Time used: ~25 min interactive audit + ~25 min write-up

---

## Addendum 2026-05-19 — operator-driven OAuth + authenticated-session audit

This is what the original audit could not catch from a Playwright-only
run: the user (operator) signed in with Google in their real browser,
then walked me through what they saw. The original audit said the
signup page looked "polished" because I tested the static signup page
without ever completing an OAuth flow. **That was wrong.**

### Headline correction

OAuth itself succeeds — the user signs in with Google, the token round-trip
works, the sidebar shows their email. **But the authenticated session
does not propagate to the API surfaces.** Multiple downstream pages
that require a session render "Authentication required" / "API request
failed" / "Authentication required for live debates" even though the
user is signed in.

This is worse than "OAuth doesn't work." It gives users false confidence
of success then fails silently when they try to use the product.

### Reclassified P0

Promoting a new P0 finding above the homepage-CTA P0:

#### P0.2 — OAuth completes but session does not propagate to API endpoints

**Symptoms observed by the operator after a successful Google sign-in:**

| Surface | Symptom |
|---|---|
| `/landing/` after running a debate | Result panel says **"Sign in to save — Keep this debate and continue from the full transcript after you sign in"** even though they are signed in |
| `/dashboard/` (`ExecutiveSummary` component) | **"Error loading dashboard: API request failed"** |
| `/dashboard/` cost panel | **"Cost data unavailable"** |
| `/debates/` start-debate flow | **`[ERROR] Authentication required`** when submitting a debate |
| `/oracle/` Oracle live-debate section | **"Authentication required for live debates. Log in to continue"** at the bottom |

**Diagnosis (via source-code reading + live curl):**

1. **The backend rejects authenticated calls to /api/v1/* with 401.**
   Direct probe: `curl https://api.aragora.ai/api/v1/usage/summary?period=month`
   returns `{"error": "Authentication required", "code": "auth_required"}`.
   This is the endpoint `aragora/live/src/components/dashboard/ExecutiveSummary.tsx`
   calls via `useSWRFetch` after the user has signed in.

2. **The frontend has the right wiring on paper.** `aragora/live/src/hooks/useSWRFetch.ts`
   reads the token from `localStorage['aragora_tokens']` and attaches
   `Authorization: Bearer <token>` to internal-API requests. The URL
   construction is correct (resolves to `https://api.aragora.ai/api/v1/...`
   not `https://aragora.ai/api/v1/...`).

3. **But the error is hidden behind a generic message.** When the API
   returns 401, `useSWRFetch.ts:69-72` throws a hardcoded
   `'API request failed'`. The 401 status is captured on the error
   object as `error.status` but never surfaced. `ExecutiveSummary`
   renders `Error loading dashboard: {error.message}`, producing the
   useless "API request failed" the user sees.

**Two distinct bugs that compound:**

- **Bug A (backend or token-pipeline):** the OAuth-issued token isn't
  being accepted by `/api/v1/*` endpoints. Either (a) the token isn't
  reaching the backend on these requests, (b) the backend's auth
  middleware doesn't recognize OAuth-issued tokens for the v1 API, or
  (c) the token is valid for `/api/auth/me` (which the sidebar uses to
  populate the user email — proven by the sidebar showing the operator's
  signed-in email) but rejected for `/api/v1/*` because of a
  scope, audience, or middleware mismatch.

- **Bug B (frontend UX):** `useSWRFetch.ts` throws
  `new Error('API request failed')` regardless of HTTP status. When a
  401 happens (which is recoverable — sign in again) the user sees the
  same opaque message as a 500 (which is not recoverable). Components
  that render `error.message` cannot distinguish auth-recoverable from
  infra-broken.

**Smallest fixes:**

- **Bug B (frontend, ~5 min):** in `useSWRFetch.ts:69-72`, set
  `error.message` based on `response.status`:
  ```ts
  const error = new Error(
    response.status === 401 ? 'Please sign in again' :
    response.status === 403 ? 'Permission denied' :
    response.status === 404 ? 'Endpoint not found' :
    response.status >= 500 ? `Server error (${response.status})` :
    `Request failed (${response.status})`
  ) as Error & { status: number };
  error.status = response.status;
  ```
  This alone wouldn't fix the underlying auth bug but would surface it.

- **Bug A (backend, scope unknown):** requires diagnosing why the
  backend rejects OAuth-issued tokens for `/api/v1/*`. Suggested
  investigation: capture the actual `Authorization: Bearer ...` header
  the dashboard request is sending in the signed-in session, compare
  against what `/api/auth/me` is sending (which works), and trace the
  middleware that validates tokens on v1 endpoints to see where it
  rejects. Likely a few hours of focused work.

**Why this is P0 above the homepage-CTA P0.1:** the homepage-CTA bug
breaks a stranger's first click. This bug breaks an authenticated
user's entire experience after sign-in. A stranger who fights through
P0.1 and signs in will then run into "Authentication required"
everywhere and conclude the product is broken.

### Additional findings from the signed-in walkthrough

- **0% confidence on a real debate result.** The operator ran "what is
  the best kind of soup?" and got `0% confidence · 3 agents · 1 round
  · 10.164s`. A completed debate with real model responses shouldn't
  return 0% confidence. Either the confidence calculation defaults to
  zero on simple questions, or the metric isn't populated and the UI
  shows a default. Adds to P1 as P1.6.

- **Receipt artifact is shown as a small unbranded subscript line.**
  The result panel shows `1582cac153de49f0…2026-05-20T03:15:38.004727+00:00`
  in fine print with no framing. Visually identical to a timestamp.
  Confirms the original audit's P1.2 and P3.1 — even with a working
  result, the differentiated artifact is invisible.

- **/landing/ doesn't react to authenticated state.** Running a debate
  while signed in still shows "Sign in to save — Keep this debate and
  continue from the full transcript after you sign in." The landing
  page's auth-state check is wrong (either reading a different storage
  key, running before `setTokens` completes, or hardcoding the prompt
  irrespective of auth state).

- **The app shell DOES work post-OAuth.** Sidebar appears with Home /
  Dashboard / Debates / Oracle / Receipts / Get Started / Settings.
  Email is shown. So the `(app)` route group correctly switches to
  authenticated rendering. The break is in what those pages can
  actually do once you reach them.

### Updated priority table

| P | Count | Examples |
|---|---:|---|
| **P0** | **2** | (NEW) OAuth completes but session doesn't propagate to /api/v1/* — dashboard, debates, oracle all show auth-required while signed in · Homepage "Try a Demo Debate" → "Debate not found" |
| **P1** | 6 | (NEW) 0% confidence on real debate results · Docs nav CORS-broken · receipt invisible on /demo/ · live-bridge always empty · /try jargon · wasteful spectate polling |
| **P2** | 3 | (unchanged) Three navbars · disabled Start button · redundant CTAs |
| **P3** | 3 | (unchanged) No "verify hash" UX · no vendor confirmation · no "what just happened" walkthrough |

### What I want to investigate next (for the operator's call)

To pin down Bug A — whether the token isn't being sent, or sent and
rejected — the operator could either:

1. **Open dev tools Network tab in their signed-in browser, navigate
   to `/dashboard/`, find the failing request to
   `api.aragora.ai/api/v1/usage/summary`**, and copy:
   - Whether the `Authorization: Bearer ...` header is present
   - The exact response status and body
   - The cookies sent

2. **Or:** inspect backend auth logs through the approved production
   access path for the matching 401 to see what the backend logged
   about the rejected token.

Without one of those, I can't tell whether the bug is in the
frontend's token retrieval (race / wrong key / stale token) or in the
backend's token validation (wrong audience / scope / middleware
ordering).

### Time used for this addendum

~30 min source-code investigation + ~15 min write-up.
