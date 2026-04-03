# Landing Page Debate UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the landing page debate experience reliable (no timeouts), intelligent (frontier model disambiguation), honest (real streaming progress), and concise (TL;DR result card).

**Architecture:** Replace regex-based `landingPreflight.ts` with a backend `/assess` endpoint calling a frontier model. Harden timeout/fallback chain. Add `tldr` synthesis post-debate. Build compact result card with clickable agent chips. Wire real spectate WebSocket events to progress UI.

**Tech Stack:** TypeScript/React (Next.js), Python (FastAPI-style handlers), Anthropic/OpenRouter APIs, WebSocket (existing spectate infrastructure)

**Spec:** `docs/superpowers/specs/2026-04-03-landing-debate-ux-design.md`

---

## File Structure

### Files to Create
| File | Responsibility |
|------|---------------|
| `aragora/live/src/components/landing/CompactDebateResult.tsx` | Compact inline result card for landing page |
| `aragora/live/src/hooks/useLandingDebateProgress.ts` | WebSocket hook for real streaming progress |
| `tests/server/handlers/test_playground_assess.py` | Tests for `/assess` endpoint |
| `tests/server/handlers/test_playground_tldr.py` | Tests for TL;DR synthesis |

### Files to Modify
| File | Changes |
|------|---------|
| `aragora/server/handlers/playground.py` | Add `_handle_assess()`, `_synthesize_tldr()`, raise timeouts, harden fallback |
| `aragora/live/src/components/landing/HeroSection.tsx` | Client timeout, replace preflight with `/assess` call, use `CompactDebateResult`, wire streaming progress |
| `aragora/live/src/components/DebateResultPreview.tsx` | Add TL;DR card at top, interpretation line, wider layout |
| `aragora/live/src/app/(standalone)/debate/[[...id]]/` | Wider max-width |

### Files to Delete
| File | Reason |
|------|--------|
| `aragora/live/src/components/landing/landingPreflight.ts` | Replaced by model-based `/assess` endpoint |
| `aragora/live/src/components/landing/__tests__/landingPreflight.test.ts` | Tests for deleted file |

### Files to Deprecate (not delete yet)
| File | Reason |
|------|--------|
| `aragora/live/src/components/LandingPage.tsx` | Non-canonical duplicate; useful logic already in HeroSection |

---

### Task 1: Consolidate Landing Architecture

**Files:**
- Inspect: `aragora/live/src/components/LandingPage.tsx` (1177 lines, non-canonical)
- Inspect: `aragora/live/src/components/landing/LandingPage.tsx` (34 lines, canonical wrapper)
- Modify: `aragora/live/src/components/LandingPage.tsx`

- [ ] **Step 1: Verify HeroSection already has all debate logic**

Check that `HeroSection.tsx` already contains: `runDebate`, `executeDebate`, `prepareLandingDebate` integration, `pendingPreflight` state, preflight option card UI (lines 632-732), `handleWrongAnswer`, result rendering, demo debate flow.

Run: `grep -n "executeDebate\|prepareLandingDebate\|pendingPreflight\|handleWrongAnswer" aragora/live/src/components/landing/HeroSection.tsx | head -20`

- [ ] **Step 2: Identify any logic in non-canonical LandingPage.tsx NOT in HeroSection**

Compare the two files. The non-canonical file (1177 lines) may have logic that HeroSection doesn't. If so, note it for merging. If HeroSection is already the superset, the non-canonical file is purely dead code.

Run: `grep -n "def\|function\|const.*=.*=>" aragora/live/src/components/LandingPage.tsx | head -30`

- [ ] **Step 3: Add deprecation banner to non-canonical LandingPage.tsx**

```typescript
/**
 * @deprecated Use components/landing/LandingPage.tsx + HeroSection.tsx instead.
 * This file is the non-canonical landing page with duplicate debate logic.
 * All active debate flow lives in HeroSection.tsx.
 * TODO: Remove once confirmed no routes import this file.
 */
```

- [ ] **Step 4: Verify no routes import the non-canonical file**

Run: `grep -rn "from.*components/LandingPage\|import.*components/LandingPage" aragora/live/src/ --include="*.ts" --include="*.tsx" | grep -v "__tests__" | grep -v "node_modules"`

If routes import it, update them to use the canonical path. If nothing imports it, it's safe to deprecate.

- [ ] **Step 5: Commit**

```bash
git add aragora/live/src/components/LandingPage.tsx
git commit -m "chore: deprecate non-canonical LandingPage.tsx in favor of landing/HeroSection"
```

---

### Task 2: Timeout & Reliability

**Files:**
- Modify: `aragora/server/handlers/playground.py:505,2508-2509,2986`
- Modify: `aragora/live/src/components/landing/HeroSection.tsx:294-311`
- Test: `tests/server/handlers/test_playground.py`

- [ ] **Step 1: Write test for broadened fallback catch**

```python
# tests/server/handlers/test_playground_assess.py (or append to test_playground.py)
import pytest

def test_run_debate_catches_all_exceptions_and_returns_mock(playground_handler, monkeypatch):
    """_run_debate must never raise — always return a mock result on failure."""
    monkeypatch.setattr(
        playground_handler, "_try_oracle_tentacles",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("simulated")),
    )
    monkeypatch.setattr(
        playground_handler, "_run_live_debate",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("simulated")),
    )
    result = playground_handler._run_debate("test topic", 2, 3, question="test")
    assert result is not None
    # Should contain mock_fallback annotation
    import json
    data = json.loads(result.body)
    assert data.get("mock_fallback") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/server/handlers/test_playground_assess.py::test_run_debate_catches_all_exceptions_and_returns_mock -v`
Expected: FAIL (ConnectionError not caught)

- [ ] **Step 3: Raise timeout constants**

In `aragora/server/handlers/playground.py`:

Line 505: Change `_ORACLE_CALL_TIMEOUT = 25.0` to `_ORACLE_CALL_TIMEOUT = 90.0`

Line 2986: Change `_LIVE_TIMEOUT = 15` to `_LIVE_TIMEOUT = 90`

- [ ] **Step 4: Broaden fallback catch in `_run_debate`**

At line 2508-2509, change:
```python
except (TimeoutError, ValueError, RuntimeError, OSError) as exc:
    logger.warning("Live debate failed, falling back to mock: %s", exc)
```
to:
```python
except Exception as exc:  # noqa: BLE001 — landing page must never error
    logger.warning("Live debate failed, falling back to mock: %s", exc)
```

Also wrap the entire oracle/tentacles path (lines 2427-2497) in a similar broad catch that falls through to live debate, so no single path can prevent the mock fallback from running.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/server/handlers/test_playground_assess.py::test_run_debate_catches_all_exceptions_and_returns_mock -v`
Expected: PASS

- [ ] **Step 6: Add client-side timeout to HeroSection**

In `aragora/live/src/components/landing/HeroSection.tsx`, in the `executeDebate` function (around line 294), add a timeout to the AbortController:

```typescript
const controller = new AbortController();
abortRef.current = controller;
const timeoutId = setTimeout(() => controller.abort(), 180_000); // 3 minutes

try {
  const res = await fetch(playgroundDebateUrl, {
    // ... existing config
    signal: controller.signal,
  });
  // ... existing response handling
} catch (err: unknown) {
  if (err instanceof Error && err.name === 'AbortError') {
    setError('The debate is taking longer than expected. Please try a shorter question or try again.');
    return;
  }
  setError('Could not connect to the server. Check your connection and try again.');
} finally {
  clearTimeout(timeoutId);
  setIsRunning(false);
}
```

- [ ] **Step 7: Verify TypeScript compiles**

Run: `cd aragora/live && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add aragora/server/handlers/playground.py aragora/live/src/components/landing/HeroSection.tsx tests/
git commit -m "fix(landing): raise timeouts to 90s backend / 180s client, harden fallback to catch all exceptions"
```

---

### Task 3: TL;DR Synthesis

**Files:**
- Modify: `aragora/server/handlers/playground.py:2410-2522`
- Create: `tests/server/handlers/test_playground_tldr.py`

- [ ] **Step 1: Write test for TL;DR synthesis**

```python
# tests/server/handlers/test_playground_tldr.py
import json
import pytest
from unittest.mock import AsyncMock, patch

def test_synthesize_tldr_returns_short_sentence():
    """_synthesize_tldr should return a single-sentence summary."""
    from aragora.server.handlers.playground import PlaygroundHandler

    handler = PlaygroundHandler.__new__(PlaygroundHandler)
    proposals = {
        "claude": "Long proposal about microwave safety with many details...",
        "gpt": "Another long proposal covering different angles...",
    }
    # Mock the agent call to return a known tldr
    with patch.object(handler, "_call_frontier_model", return_value="Yes, reheating chicken nuggets in a microwave is safe."):
        result = handler._synthesize_tldr("Can I microwave chicken nuggets?", proposals)

    assert result is not None
    assert len(result) < 300
    assert result.endswith(".")

def test_synthesize_tldr_returns_fallback_on_timeout():
    """On timeout, _synthesize_tldr should truncate final_answer."""
    from aragora.server.handlers.playground import PlaygroundHandler

    handler = PlaygroundHandler.__new__(PlaygroundHandler)
    proposals = {"claude": "First sentence here. Second sentence with more detail."}

    with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError("timeout")):
        result = handler._synthesize_tldr(
            "test question",
            proposals,
            fallback_text="First sentence here. Second sentence with more detail.",
        )

    assert result == "First sentence here."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/server/handlers/test_playground_tldr.py -v`
Expected: FAIL (`_synthesize_tldr` doesn't exist)

- [ ] **Step 3: Implement `_synthesize_tldr` method**

Add to `PlaygroundHandler` class in `playground.py`:

```python
def _synthesize_tldr(
    self,
    question: str,
    proposals: dict[str, str],
    fallback_text: str | None = None,
) -> str:
    """Synthesize a one-sentence TL;DR from agent proposals using a frontier model."""
    prompt = (
        "Given these agent proposals responding to the question below, "
        "write a single-sentence direct answer. Be practical, not philosophical. "
        "Do not mention the agents or the debate process.\n\n"
        f"Question: {question}\n\n"
    )
    for agent, text in proposals.items():
        prompt += f"{agent}: {text[:500]}\n\n"
    prompt += "One-sentence answer:"

    try:
        return self._call_frontier_model(prompt, timeout=5.0)
    except (TimeoutError, ConnectionError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("TL;DR synthesis failed, using fallback: %s", exc)
        if fallback_text:
            # Return first sentence
            first_dot = fallback_text.find(". ")
            if first_dot > 0:
                return fallback_text[: first_dot + 1]
            return fallback_text[:200]
        return ""
```

- [ ] **Step 4: Implement `_call_frontier_model` helper**

```python
def _call_frontier_model(self, prompt: str, timeout: float = 5.0) -> str:
    """Call the fastest available frontier model for a short generation task."""
    import asyncio
    from aragora.agents.api_agents.anthropic import AnthropicAgent

    async def _run() -> str:
        try:
            agent = AnthropicAgent(model="claude-sonnet-4-20250514")
            response = await asyncio.wait_for(
                agent.generate(prompt, max_tokens=200),
                timeout=timeout,
            )
            return response.strip()
        except ImportError:
            from aragora.agents.api_agents.openrouter import OpenRouterAgent
            agent = OpenRouterAgent(model="anthropic/claude-sonnet-4-20250514")
            response = await asyncio.wait_for(
                agent.generate(prompt, max_tokens=200),
                timeout=timeout,
            )
            return response.strip()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
```

- [ ] **Step 5: Wire `_synthesize_tldr` into `_run_debate`**

In `_run_debate()`, after the result dict is assembled but before `json_response()`:

Find the lines where the result dict is built (around lines 2460-2470 for oracle, 2490-2497 for tentacles, 2505-2507 for live). In each success path, before the `return self._persist_and_respond(json_response(...))` call, add:

```python
# Synthesize TL;DR
proposals = result_dict.get("proposals", {})
if proposals:
    result_dict["tldr"] = self._synthesize_tldr(
        question or topic,
        proposals,
        fallback_text=result_dict.get("final_answer", ""),
    )
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/server/handlers/test_playground_tldr.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add aragora/server/handlers/playground.py tests/server/handlers/test_playground_tldr.py
git commit -m "feat(landing): add TL;DR synthesis via frontier model after debate completion"
```

---

### Task 4: Replace Regex Disambiguation with Frontier Model `/assess`

**Files:**
- Modify: `aragora/server/handlers/playground.py:2229-2238`
- Modify: `aragora/live/src/components/landing/HeroSection.tsx:366-386`
- Delete: `aragora/live/src/components/landing/landingPreflight.ts`
- Delete: `aragora/live/src/components/landing/__tests__/landingPreflight.test.ts`
- Create: `tests/server/handlers/test_playground_assess.py`

- [ ] **Step 1: Write test for `/assess` endpoint**

```python
# tests/server/handlers/test_playground_assess.py
import json
import pytest
from unittest.mock import patch

def test_assess_clear_question_returns_ready(playground_handler, mock_handler):
    """Clear questions should return type=ready with no options."""
    mock_handler.body = json.dumps({"question": "Should we use React or Vue for our frontend?"}).encode()

    with patch.object(playground_handler, "_call_frontier_model", return_value=json.dumps({
        "clear": True,
        "topic": "Should we use React or Vue for our frontend?"
    })):
        result = playground_handler._handle_assess(mock_handler)

    data = json.loads(result.body)
    assert data["type"] == "ready"
    assert data["option"]["debatePrompt"] == "Should we use React or Vue for our frontend?"

def test_assess_ambiguous_question_returns_confirm(playground_handler, mock_handler):
    """Ambiguous questions should return type=confirm with interpretation options."""
    mock_handler.body = json.dumps({"question": "should i cook my chickens in a microwave"}).encode()

    with patch.object(playground_handler, "_call_frontier_model", return_value=json.dumps({
        "clear": False,
        "interpretations": [
            "Is it safe to reheat pre-cooked chicken in a microwave?",
            "What are the ethical considerations of factory-farmed chicken?",
            "How should I cook raw chicken safely?"
        ]
    })):
        result = playground_handler._handle_assess(mock_handler)

    data = json.loads(result.body)
    assert data["type"] == "confirm"
    assert len(data["preflight"]["options"]) >= 3

def test_assess_timeout_returns_ready(playground_handler, mock_handler):
    """On timeout, /assess should return ready (never block)."""
    mock_handler.body = json.dumps({"question": "test"}).encode()

    with patch.object(playground_handler, "_call_frontier_model", side_effect=TimeoutError):
        result = playground_handler._handle_assess(mock_handler)

    data = json.loads(result.body)
    assert data["type"] == "ready"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/server/handlers/test_playground_assess.py -v`
Expected: FAIL (`_handle_assess` doesn't exist)

- [ ] **Step 3: Implement `_handle_assess` in PlaygroundHandler**

Add route dispatch in `handle_post()` before line 2238:
```python
if path == "/api/v1/playground/assess":
    return self._handle_assess(handler)
```

Add the method:
```python
def _handle_assess(self, handler: Any) -> HandlerResult:
    """Assess question ambiguity using a frontier model."""
    body = json.loads(handler.body or b"{}")
    question = str(body.get("question", "")).strip()
    if not question:
        return json_response({"type": "ready", "option": self._build_ready_option(question)})

    # Rate limit: 10 per 60s
    rate_check = self._check_rate_limit(handler, window=60, max_requests=10, key_suffix="assess")
    if rate_check is not None:
        return rate_check

    prompt = (
        "You are an question-assessment system. Analyze this user question and determine if it is "
        "clear enough to debate directly, or if it could be interpreted multiple ways.\n\n"
        f"Question: {question}\n\n"
        "Respond with JSON only:\n"
        '- If clear: {"clear": true, "topic": "<the question as-is>"}\n'
        '- If ambiguous: {"clear": false, "interpretations": ["interpretation 1", "interpretation 2", "interpretation 3"]}\n'
        "JSON response:"
    )

    try:
        raw = self._call_frontier_model(prompt, timeout=5.0)
        parsed = json.loads(raw)
    except (TimeoutError, ConnectionError, json.JSONDecodeError, Exception) as exc:
        logger.debug("Assess call failed, returning ready: %s", exc)
        return json_response({"type": "ready", "option": self._build_ready_option(question)})

    if parsed.get("clear", True):
        topic = parsed.get("topic", question)
        return json_response({"type": "ready", "option": self._build_ready_option(topic)})

    # Build preflight options from interpretations
    interpretations = parsed.get("interpretations", [])
    options = []
    for i, interp in enumerate(interpretations[:4]):
        options.append({
            "id": f"interp-{i}",
            "label": interp[:80],
            "description": interp,
            "originalQuestion": question,
            "interpretedQuestion": interp,
            "debatePrompt": interp,
            "agents": 3,
            "rounds": 2,
            "recommended": i == 0,
        })
    # Always include "use original wording" as last option
    options.append({
        "id": "original",
        "label": "Use original wording",
        "description": "Debate the question exactly as written.",
        "originalQuestion": question,
        "interpretedQuestion": question,
        "debatePrompt": question,
        "agents": 3,
        "rounds": 2,
    })

    return json_response({
        "type": "confirm",
        "preflight": {
            "title": "This question could mean a few things",
            "prompt": "Pick the interpretation you want Aragora to debate.",
            "options": options,
        },
    })

def _build_ready_option(self, question: str) -> dict:
    return {
        "id": "original",
        "label": "Use original wording",
        "description": question,
        "originalQuestion": question,
        "interpretedQuestion": question,
        "debatePrompt": question,
        "agents": 3,
        "rounds": 2,
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/server/handlers/test_playground_assess.py -v`
Expected: PASS

- [ ] **Step 5: Replace `prepareLandingDebate` with `/assess` API call in HeroSection**

In `HeroSection.tsx`, replace the `runDebate` function (lines 366-386):

```typescript
async function runDebate(rawQuestion: string) {
  setError(null);
  setEditorNotice(null);
  setResult(null);
  setLastTopic(rawQuestion);
  setIsRunning(true);

  try {
    // Call /assess to check ambiguity (frontier model, not regex)
    const assessRes = await fetch(
      `${backendBase}/api/v1/playground/assess`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: rawQuestion }),
        signal: AbortSignal.timeout(8000), // 8s max for assess
      },
    );

    if (!assessRes.ok) {
      // Assess failed — debate raw question directly
      void executeDebate({
        id: 'original',
        label: rawQuestion,
        description: rawQuestion,
        originalQuestion: rawQuestion,
        interpretedQuestion: rawQuestion,
        debatePrompt: rawQuestion,
        agents: 3,
        rounds: 2,
      });
      return;
    }

    const assessment = await assessRes.json();

    if (assessment.type === 'confirm') {
      setPendingPreflight(assessment.preflight);
      setIsRunning(false);
      trackEvent('preflight_shown', {
        option_count: assessment.preflight.options.length,
        question_length: rawQuestion.length,
      });
      return;
    }

    // Clear — proceed directly
    setPendingPreflight(null);
    void executeDebate(assessment.option);
  } catch {
    // Assess call failed entirely — debate raw question
    setIsRunning(false);
    void executeDebate({
      id: 'original',
      label: rawQuestion,
      description: rawQuestion,
      originalQuestion: rawQuestion,
      interpretedQuestion: rawQuestion,
      debatePrompt: rawQuestion,
      agents: 3,
      rounds: 2,
    });
  }
}
```

- [ ] **Step 6: Remove `prepareLandingDebate` import from HeroSection**

Remove the import line: `import { prepareLandingDebate } from './landingPreflight';`

- [ ] **Step 7: Delete landingPreflight.ts and its test**

```bash
rm aragora/live/src/components/landing/landingPreflight.ts
rm aragora/live/src/components/landing/__tests__/landingPreflight.test.ts
```

- [ ] **Step 8: Verify TypeScript compiles**

Run: `cd aragora/live && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors (may need to remove other imports of `prepareLandingDebate` or `LandingPreparedDebateOption`)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(landing): replace regex disambiguation with frontier model /assess endpoint"
```

---

### Task 5: Compact Inline Result Card

**Files:**
- Create: `aragora/live/src/components/landing/CompactDebateResult.tsx`
- Modify: `aragora/live/src/components/landing/HeroSection.tsx`

- [ ] **Step 1: Create `CompactDebateResult` component**

Create `aragora/live/src/components/landing/CompactDebateResult.tsx`:

```typescript
'use client';

import { useState } from 'react';
import Link from 'next/link';
import type { DebateResponse } from '../DebateResultPreview';

// Agent color mapping (reuse from DebateResultPreview)
const AGENT_COLORS: Record<string, string> = {
  claude: '#0369a1',
  gpt: '#92400e',
  grok: '#9d174d',
  gemini: '#7c3aed',
  mistral: '#0f766e',
  deepseek: '#dc2626',
};

function chipColor(name: string): string {
  const lower = name.toLowerCase();
  for (const [key, color] of Object.entries(AGENT_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return '#6b7280';
}

function chipBg(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('claude')) return '#e0f2fe';
  if (lower.includes('gpt')) return '#fef3c7';
  if (lower.includes('grok')) return '#fce7f3';
  if (lower.includes('gemini')) return '#ede9fe';
  if (lower.includes('mistral')) return '#ccfbf1';
  if (lower.includes('deepseek')) return '#fee2e2';
  return '#f3f4f6';
}

function stripMarkdown(text: string): string {
  return text
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[(.*?)\]\(.*?\)/g, '$1')
    .replace(/\n+/g, ' ')
    .trim();
}

interface CompactDebateResultProps {
  result: DebateResponse;
  onWrongAnswer?: (result: DebateResponse) => void;
  onShare?: (result: DebateResponse) => void;
}

export function CompactDebateResult({ result, onWrongAnswer, onShare }: CompactDebateResultProps) {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const tldr = (result as Record<string, unknown>).tldr as string | undefined;
  const interpretedQuestion = (result as Record<string, unknown>).interpreted_question as string | undefined;
  const originalQuestion = (result as Record<string, unknown>).original_question as string | undefined;
  const showInterpretation = interpretedQuestion && originalQuestion && interpretedQuestion !== originalQuestion;

  const proposalEntries = Object.entries(result.proposals);
  const pct = Math.round(result.confidence * 100);

  const handleShare = async () => {
    const url = `${window.location.origin}/debate/${result.id}`;
    try { await navigator.clipboard.writeText(url); } catch { /* ignore */ }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    onShare?.(result);
  };

  return (
    <div className="space-y-4 text-left" style={{ fontFamily: 'var(--font-landing)' }}>
      {/* Interpretation line */}
      {showInterpretation && (
        <p className="text-xs text-[var(--text-muted)] italic">
          Aragora interpreted this as: {interpretedQuestion}
        </p>
      )}

      {/* TL;DR answer card */}
      <div
        className="rounded-2xl border p-5"
        style={{
          borderColor: 'var(--accent)',
          backgroundColor: 'color-mix(in srgb, var(--accent) 5%, var(--surface))',
        }}
      >
        <div className="text-[11px] uppercase tracking-[0.1em] font-semibold mb-2" style={{ color: 'var(--accent)' }}>
          Aragora&apos;s Answer
        </div>
        <div className="text-base font-semibold leading-relaxed text-[var(--text)]">
          {tldr || result.final_answer?.slice(0, 200) || 'No verdict returned.'}
        </div>
      </div>

      {/* Metadata row */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 items-center text-xs text-[var(--text-muted)]">
        <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{pct}% confidence</span>
        <span>&middot;</span>
        <span>{result.participants.length} agents</span>
        <span>&middot;</span>
        <span>{result.rounds_used} round{result.rounds_used !== 1 ? 's' : ''}</span>
        <span>&middot;</span>
        <span>{result.duration_seconds}s</span>
      </div>

      {/* Agent chips — clickable to expand proposals */}
      <div className="flex flex-wrap gap-2">
        {result.participants.map((name) => (
          <button
            key={name}
            onClick={() => setExpandedAgent(expandedAgent === name ? null : name)}
            className="rounded-full text-xs font-medium px-3 py-1.5 transition-all cursor-pointer"
            style={{
              backgroundColor: expandedAgent === name ? chipColor(name) : chipBg(name),
              color: expandedAgent === name ? '#fff' : chipColor(name),
              border: `1px solid ${chipColor(name)}33`,
            }}
          >
            {name}
          </button>
        ))}
      </div>

      {/* Expanded agent proposal */}
      {expandedAgent && result.proposals[expandedAgent] && (
        <div
          className="rounded-xl border p-4 text-sm leading-relaxed text-[var(--text)]"
          style={{
            borderColor: `${chipColor(expandedAgent)}33`,
            borderLeftWidth: '3px',
            borderLeftColor: chipColor(expandedAgent),
          }}
        >
          <div className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: chipColor(expandedAgent) }}>
            {expandedAgent}
          </div>
          <p>{stripMarkdown(result.proposals[expandedAgent]).slice(0, 400)}</p>
          {result.proposals[expandedAgent].length > 400 && (
            <Link
              href={`/debate/${result.id}`}
              className="text-xs mt-2 inline-block"
              style={{ color: 'var(--accent)' }}
            >
              Read full response &rarr;
            </Link>
          )}
        </div>
      )}

      {/* Receipt row */}
      {result.receipt_hash && (
        <div className="text-xs text-[var(--text-muted)]">
          Receipt: {result.receipt_hash.slice(0, 16)}... &middot; {result.receipt?.timestamp || new Date().toISOString()}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3">
        {result.id && (
          <Link
            href={`/debate/${result.id}`}
            className="text-sm font-semibold transition-opacity hover:opacity-80"
            style={{ color: 'var(--accent)' }}
          >
            View full debate &rarr;
          </Link>
        )}
        <button
          onClick={handleShare}
          className="text-xs text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors cursor-pointer"
        >
          {copied ? 'Copied!' : 'Share'}
        </button>
        {onWrongAnswer && (
          <button
            onClick={() => onWrongAnswer(result)}
            className="text-xs text-[var(--text-muted)] hover:text-[var(--crimson,#dc2626)] transition-colors cursor-pointer ml-auto"
          >
            Wrong answer?
          </button>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire `CompactDebateResult` into HeroSection**

In `HeroSection.tsx`, replace the `DebateResultPreview` usage (around line 935-948) with `CompactDebateResult`:

```typescript
import { CompactDebateResult } from './CompactDebateResult';

// In the JSX, replace <DebateResultPreview ... /> with:
{result && (
  <div ref={resultRef}>
    <CompactDebateResult
      result={result}
      onWrongAnswer={handleWrongAnswer}
      onShare={(debateResult) => {
        trackEvent('share_clicked', {
          result_mode: debateResult.result_mode || 'full',
        });
      }}
    />
  </div>
)}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd aragora/live && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add aragora/live/src/components/landing/CompactDebateResult.tsx aragora/live/src/components/landing/HeroSection.tsx
git commit -m "feat(landing): add CompactDebateResult with clickable agent chips and TL;DR card"
```

---

### Task 6: Real Streaming Progress

**Files:**
- Create: `aragora/live/src/hooks/useLandingDebateProgress.ts`
- Modify: `aragora/live/src/components/landing/HeroSection.tsx`
- Modify: `aragora/server/handlers/playground.py`

- [ ] **Step 1: Create `useLandingDebateProgress` hook**

Create `aragora/live/src/hooks/useLandingDebateProgress.ts`:

```typescript
import { useCallback, useEffect, useRef, useState } from 'react';

export interface DebateProgressEvent {
  phase: 'assessing' | 'starting' | 'proposing' | 'critiquing' | 'voting' | 'consensus' | 'done';
  agent?: string;
  round?: number;
  totalRounds?: number;
  content?: string; // streaming proposal text
}

interface UseLandingDebateProgressOptions {
  debateId: string | null;
  wsUrl: string;
  enabled: boolean;
}

export function useLandingDebateProgress({ debateId, wsUrl, enabled }: UseLandingDebateProgressOptions) {
  const [events, setEvents] = useState<DebateProgressEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const startTime = useRef<number>(Date.now());
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Elapsed timer
  useEffect(() => {
    if (!enabled) return;
    startTime.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime.current) / 1000));
    }, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [enabled]);

  // WebSocket connection
  useEffect(() => {
    if (!enabled || !debateId) return;

    try {
      const ws = new WebSocket(`${wsUrl}?debate_id=${debateId}`);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const mapped = mapEventToProgress(data);
          if (mapped) {
            setEvents((prev) => [...prev, mapped]);
          }
        } catch { /* ignore parse errors */ }
      };

      return () => { ws.close(); };
    } catch {
      // WebSocket not available — fallback to no streaming
      return;
    }
  }, [enabled, debateId, wsUrl]);

  const reset = useCallback(() => {
    setEvents([]);
    setConnected(false);
    setElapsed(0);
  }, []);

  const latestEvent = events.length > 0 ? events[events.length - 1] : null;

  return { events, latestEvent, connected, elapsed, reset };
}

function mapEventToProgress(data: Record<string, unknown>): DebateProgressEvent | null {
  const type = data.type || data.event_type;

  switch (type) {
    case 'debate_start':
      return { phase: 'starting', agent: data.agents as string | undefined };
    case 'agent_message':
    case 'proposal':
      return {
        phase: 'proposing',
        agent: (data.agent || data.agent_name) as string | undefined,
        round: data.round as number | undefined,
        content: data.content as string | undefined,
      };
    case 'critique':
      return {
        phase: 'critiquing',
        agent: (data.agent || data.agent_name) as string | undefined,
        round: data.round as number | undefined,
      };
    case 'vote':
      return { phase: 'voting', agent: (data.agent || data.agent_name) as string | undefined };
    case 'consensus':
      return { phase: 'consensus' };
    case 'debate_end':
      return { phase: 'done' };
    default:
      return null;
  }
}
```

- [ ] **Step 2: Accept `debateId` in playground request body**

In `playground.py`, in the debate request body parsing (around line 2240-2260), add:

```python
debate_id = body.get("debate_id") or str(uuid.uuid4())[:16]
```

Pass this `debate_id` through to `_run_debate` and use it when creating the arena/debate.

- [ ] **Step 3: Replace fake progress UI in HeroSection**

In `HeroSection.tsx`, replace the fake phase progression (lines 465-574) with real streaming:

```typescript
import { useLandingDebateProgress } from '@/hooks/useLandingDebateProgress';
import { v4 as uuidv4 } from 'uuid'; // or use crypto.randomUUID()

// In component body:
const [debateId, setDebateId] = useState<string | null>(null);
const wsUrl = backendBase.replace(/^http/, 'ws') + '/ws/spectate';
const progress = useLandingDebateProgress({
  debateId,
  wsUrl,
  enabled: isRunning,
});

// In executeDebate, before the fetch:
const nextDebateId = crypto.randomUUID().slice(0, 16);
setDebateId(nextDebateId);
// Add debate_id to the request body

// Replace fake phase UI with:
{isRunning && (
  <div className="mt-6 max-w-xl mx-auto p-5 rounded-2xl border border-[var(--border)] bg-[var(--surface)]">
    <div className="flex items-center gap-3 mb-3">
      <div className="w-3 h-3 rounded-full bg-[var(--accent)] animate-pulse" />
      <span className="text-sm font-medium text-[var(--text)]">
        {progress.latestEvent?.phase === 'proposing' && progress.latestEvent.agent
          ? `${progress.latestEvent.agent} is responding...`
          : progress.latestEvent?.phase === 'critiquing'
            ? `Round ${progress.latestEvent.round || 1}: Critiques...`
            : progress.latestEvent?.phase === 'voting'
              ? 'Building consensus...'
              : 'Asking agents...'}
      </span>
      <span className="ml-auto text-xs text-[var(--text-muted)]">{progress.elapsed}s</span>
    </div>
    {/* Show streaming content if available */}
    {progress.latestEvent?.content && (
      <div className="text-xs text-[var(--text-muted)] leading-relaxed mt-2 max-h-24 overflow-hidden">
        {progress.latestEvent.content.slice(0, 300)}...
      </div>
    )}
  </div>
)}
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd aragora/live && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add aragora/live/src/hooks/useLandingDebateProgress.ts aragora/live/src/components/landing/HeroSection.tsx aragora/server/handlers/playground.py
git commit -m "feat(landing): real streaming progress via spectate WebSocket replacing fake phases"
```

---

### Task 7: Full Debate Page Improvements

**Files:**
- Modify: `aragora/live/src/components/DebateResultPreview.tsx`
- Modify: `aragora/live/src/app/(standalone)/debate/[[...id]]/`

- [ ] **Step 1: Add TL;DR card to DebateResultPreview**

In `DebateResultPreview.tsx`, add a TL;DR card after the summary bar (around line 257) and before the proposals section:

```tsx
{/* TL;DR answer */}
{(result as Record<string, unknown>).tldr && (
  <div
    className="rounded-2xl border p-5"
    style={{
      borderColor: 'var(--accent)',
      backgroundColor: 'color-mix(in srgb, var(--accent) 5%, var(--surface))',
    }}
  >
    <div className="text-[11px] uppercase tracking-[0.1em] font-semibold mb-2" style={{ color: 'var(--accent)' }}>
      Summary
    </div>
    <div className="text-base font-semibold leading-relaxed text-[var(--text)]">
      {(result as Record<string, unknown>).tldr as string}
    </div>
  </div>
)}
```

- [ ] **Step 2: Add interpretation line**

Before the TL;DR card, add:

```tsx
{/* Interpretation notice */}
{(result as Record<string, unknown>).interpreted_question &&
  (result as Record<string, unknown>).interpreted_question !== ((result as Record<string, unknown>).original_question || result.topic) && (
  <p className="text-xs text-[var(--text-muted)] italic">
    Aragora interpreted this as: {(result as Record<string, unknown>).interpreted_question as string}
  </p>
)}
```

- [ ] **Step 3: Widen debate page layout**

Find the layout container in `aragora/live/src/app/(standalone)/debate/[[...id]]/` and increase max-width from ~800px to 960px:

```tsx
<div className="mx-auto" style={{ maxWidth: '960px' }}>
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd aragora/live && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add aragora/live/src/components/DebateResultPreview.tsx aragora/live/src/app/
git commit -m "feat(debate): add TL;DR card, interpretation line, and wider layout to full debate page"
```

---

## Implementation Order Summary

| Order | Task | Estimated Effort | Dependencies |
|-------|------|-----------------|--------------|
| 1 | Consolidate landing architecture | 15 min | None |
| 2 | Timeout & reliability | 30 min | None |
| 3 | TL;DR synthesis | 45 min | None (backend only) |
| 4 | Replace regex with model `/assess` | 60 min | Task 3 (reuses `_call_frontier_model`) |
| 5 | Compact result card | 45 min | Task 3 (uses `tldr` field) |
| 6 | Real streaming progress | 60 min | Task 5 (wired into HeroSection) |
| 7 | Full debate page improvements | 20 min | Task 3 (uses `tldr` field) |
