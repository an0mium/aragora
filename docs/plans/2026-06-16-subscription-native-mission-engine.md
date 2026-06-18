# Subscription-Native Mission Engine (predictable-cost path)

**Status:** proposal · **Date:** 2026-06-16 · **Stacks on:** [native mission orchestrator](2026-06-16-native-mission-orchestrator.md) (#8463)

## Goal

Make maximum autonomous progress driven **only by flat-rate subscriptions** —
OpenAI Codex (ChatGPT), Claude Pro Max, Factory Max (Droid), Gemini Max, Grok
Ultra — so cost is **predictable**. This is explicitly *not* the cost/throughput
optimum (metered API is), but predictability is the priority right now. This doc
is the subscription-tuned specialization of #8463's engine.

## Why subscriptions halt today (the four causes — none are orchestration logic)

1. **402 usage walls** — every provider's CLI hits 5-hour + weekly caps.
2. **Multi-fleet contention** — Factory mission + boss_loop + hand-driven Codex
   Desktop + Claude desktop MCP swarm, all on one machine / one account, wedge
   each other (the chroma-mcp / `claude -p` MCP hang is the canonical case).
3. **Hard-halts-without-relay** — `needs_human` / consecutive-failures stop and
   wait for a human "proceed" instead of parking + notifying + auto-resuming.
4. **MFA/secrets friction** — the merge-quorum's grok family runs via xAI API →
   AWS Secrets Manager → MFA every ~1h, forcing manual relay.

## The single biggest win: a pure-CLI, subscription-native merge quorum

The merge-quorum gate needs **2 distinct model families**. Today the default is
`claude + grok`, and grok runs via API+MFA (cause #4). But the gate already
supports the **OpenAI family via the Codex CLI**:

- `aragora/swarm/quorum_evidence.py` `_run_openai_reviewer()` uses the OpenAI API
  **only if `OPENAI_API_KEY` is set, else `codex exec`** (subscription).
- `_run_claude_cli()` uses `_claude_reviewer_command()` — `claude -p` with MCP
  **disabled** (`--strict-mcp-config --mcp-config '{"mcpServers":{}}'`, the #8421
  fix), immune to the chroma-mcp wedge.

So the subscription quorum is:

```
env -u OPENAI_API_KEY \
  collect_quorum_evidence --repo <repo> --pr <N> --reviewers claude openai --apply
```

→ `claude -p` (Anthropic, subscription) + `codex exec` (OpenAI, subscription).
**Zero API keys, zero AWS, zero MFA, zero grok.** This removes cause #4 entirely
and is the highest-leverage, lowest-risk change. (It is an *invocation* choice,
not an edit to the merge-authority gate logic — `DEFAULT_FAMILIES` in
`quorum_evidence.py` stays untouched; that file is Tier-4.)

### Grok note (verified 2026-06-16)
The installed `grok-cli` (v1.0.1, `/opt/homebrew/bin/grok`) is **not usable**: it
errors `410 "Live search is deprecated"` against the current xAI API, and it is
`GROK_API_KEY`-based (API, not the Grok Ultra subscription). **Do not depend on
grok for the gate.** Grok Ultra stays a research / red-team surface (grok.com),
not an automated reviewer, unless a working Agent-Tools-API CLI lands.

## Provider → role map (play to strengths; 5 independent rate pools)

| Subscription | Role | Access |
|---|---|---|
| **OpenAI Codex** | Primary code **writer** + OpenAI-family quorum reviewer | `codex exec --full-auto` / `codex exec` |
| **Claude Pro Max** | Orchestrator + writer + Anthropic-family quorum reviewer | `claude -p --strict-mcp-config --mcp-config '{}'` |
| **Factory Max (Droid)** | **One** structured multi-phase mission at a time | Droid app |
| **Gemini Max** | Research / 2nd-opinion review / observability / backup family | `gemini` CLI |
| **Grok Ultra** | Ad-hoc red-team / research (not a gate dependency) | grok.com |

## Machine layout: one account-fleet per Mac → zero contention

- **Mac-1 (engine):** headless `boss_loop` — codex (writer) + claude/codex
  (quorum) + drain. **Quit the Claude desktop app here** (it hosts the MCP swarm).
- **Mac-2 (mission):** Factory Droid, single coherent mission (separate Factory
  Max sub → no collision with the others).
- **Mac-3 (ad-hoc):** Gemini research/observability + occasional Grok red-team.

Five subs, three boxes, **no same-account contention** — the root cause of every
CLI wedge.

## Surviving the 402 ceiling (turns "proceed every 3 min" into "walk away")

1. **Cap-aware auto-resume** — a launchd job parses the "resets in Xh" pause and
   re-fires at reset, instead of waiting on a human "proceed". boss_loop already
   has `--max-hours` + KeepAlive; this adds the backoff. (Factory Droid is a
   desktop app — that leg stays manual; drive its work through boss_loop instead.)
2. **Stagger start times** so the 5h windows don't reset together → almost always
   ≥1 provider with budget.
3. **Drain-when-capped** — when the *writer* is rate-limited, drain the existing
   PR queue (merge-green / close-empty: `scripts/boss_drain_pass.py`) using only
   the *reviewer* subs. Useful work without the writer.
4. **Relay-with-timeout** (#8466, merged core) — on `needs_human`, PARK + notify
   (Slack) + continue other lanes; never hard-stop the whole mission.

## Sequencing (mostly wiring; cores already merged)

- **T0 — today, no code:** adopt `claude + openai(codex)` quorum (`env -u
  OPENAI_API_KEY`), MCP-disable reviewers, one account per machine, quit Claude
  desktop on Mac-1, **stop hand-driving interactive Codex Desktop**.
- **T1:** subscription evidence+settle+merge driver (`--reviewers claude openai`,
  MFA-free) + drain-when-capped (drain stack landed 2026-06-16).
- **T2:** cap-aware auto-resume daemon → eliminates the manual relay.
- **T3:** nomic free-text goal front-door (#8465, merged) → "set a goal" entry.
- **T4:** relay-with-timeout wired into boss_loop (#8466 core merged) → Slack
  park/resume.

## Honest ceiling

Subscriptions give predictable cost but a **hard weekly aggregate cap**. This
design maximizes work-per-window and removes the manual relay, but cannot exceed
the caps. Truly uninterrupted multi-day autonomy is the one thing only metered
API spend buys — a conscious trade deferred for cost predictability. The engine
is built so flipping to API later is a routing change, not a rebuild.
