# Agent Bridge — CLI-Resume Transport (Design Refresh)

Last updated: 2026-04-21
Status: design refresh, implementation-ready (pending PR review)

**Supersedes transport choice in:** [docs/plans/2026-04-19-codex-claude-supervisory-bridge.md](2026-04-19-codex-claude-supervisory-bridge.md) (deferred 2026-04-19).
**Tracks:** new issue (to be filed against this PR).

## Summary

Pivot the agent-bridge transport layer from tmux keystroke automation to the native resumable-session CLI APIs that `claude`, `codex`, and `droid` all provide. Keep the existing tmux/session-mux substrate (`scripts/agent_bridge.py`, `aragora/swarm/session_mux.py`) as a legacy human-operator surface — do **not** delete or retrofit it.

Ship in four bounded slices: design refresh (this PR), broker core, observable UI, supervisory features (only if needed).

## Why the pivot

The original 2026-04-19 plan assumed tmux pane automation as the transport: `tmux send-keys` in, `capture-pane` out, footer-contract parsing for state. That plan was deferred because the adapter layer (ANSI stripping, prompt-marker detection, TUI quirks) was estimated to be a week that takes two months.

The three harnesses that matter all now expose **native resumable-session CLI APIs** that bypass the TUI parsing problem entirely:

- `claude --resume <uuid> --print` (Claude Code)
- `codex exec resume <thread_id>` (OpenAI Codex CLI)
- `droid exec -s <session_id>` (Factory Droid CLI)

Each call is a subprocess with clean stdin/stdout. Context is persisted by the harness/CLI (via its own session storage — location varies: `~/.codex/sessions/`, `~/.factory/sessions/`, `~/.claude/projects/`), addressed by UUID/thread_id. No tmux gymnastics.

## Empirical validation (2026-04-21)

Executed a four-turn cross-harness test confirming persistent context and heterogeneous handoff:

| Turn | Harness (model) | Session | Action | Result |
|---|---|---|---|---|
| 1 | Codex (gpt-5.4) | `019db152-df99...` | Generate `TOKEN_C` | `K7M2Q9` |
| 2 | Droid (Opus 4.7) | `2beb290f-47cb...` | Generate `TOKEN_D` | `K7m2Xq` |
| 3 | Codex **resume** | same | Recall own TOKEN_C + receive droid's TOKEN_D + combine | `K7M2Q9K7m2Xq` ✅ |
| 4 | Droid **resume** | same | Recall own TOKEN_D + verify codex's combined string | `MATCH=YES` ✅ |

Commands used (reproducible):

```bash
# Turn 1
codex exec "Generate a unique random 6-character alphanumeric token. Remember it as TOKEN_C. Respond with ONLY the token."
# → K7M2Q9. Session id in stdout header: `session id: 019db152-df99-77c0-9863-08cf1a2a994f`

# Turn 2
droid exec --auto low "Generate a unique random 6-character alphanumeric token. Remember it as TOKEN_D. Respond with ONLY the token."
# → K7m2Xq. Session id via filesystem: ~/.factory/sessions/<cwd>/last.settings.json

# Turn 3
codex exec resume 019db152-df99-77c0-9863-08cf1a2a994f "What was TOKEN_C that you generated earlier? Also, Droid told me their TOKEN_D is K7m2Xq. Produce 'TOKEN_C=<your>, TOKEN_D=<droid>, COMBINED=<concat>'."
# → TOKEN_C=K7M2Q9, TOKEN_D=K7m2Xq, COMBINED=K7M2Q9K7m2Xq

# Turn 4
droid exec --auto low -s 2beb290f-47cb-47c9-a3e5-67f0f85ad0de "What was your TOKEN_D? Also codex sent this combined result: '<turn 3 output>'. Verify TOKEN_D matches. Respond: 'TOKEN_D=<your>, MATCH=YES|NO'."
# → TOKEN_D=K7m2Xq, MATCH=YES
```

Total: ~3 minutes, 4 subprocess calls, two heterogeneous model families (GPT-5.4 ↔ Claude Opus 4.7), one human orchestrator (Claude Code session).

## Architecture

### Broker module

Location: `aragora/swarm/agent_bridge/` (new).

Responsibilities:
- Own run state (list of active logical agents + their harness sessions)
- Session registry keyed by `(run_id, role)` → `{harness, session_id, worktree_path, branch}`
- Event logging (append-only JSONL)
- Per-harness dispatch via transport adapters
- Footer extraction + repair-prompt on malformed turns

### Transport interface

```python
class Transport(Protocol):
    def start_session(self, prompt: str, *, cwd: Path, model: str) -> SessionStart: ...
    def resume_turn(self, session_id: str, prompt: str) -> TurnResult: ...
    def parse_response(self, raw: str) -> ParsedTurn: ...
    def healthcheck(self) -> HealthStatus: ...
```

`SessionStart` returns both the session UUID and the first turn's response. `TurnResult` returns raw stdout/stderr + the parsed footer.

### Per-harness adapters

Three initial adapters: `claude.py`, `codex.py`, `droid.py`.

**Claude Code adapter:**
- Start: `claude -p --session-id <broker-assigned-uuid> "<prompt>"`
- Resume: `claude -p --resume <uuid> "<prompt>"`
- Broker assigns UUID. **Must be a valid UUID** (claude help: `--session-id <uuid>  Use a specific session ID ... (must be a valid UUID)`). Use `uuid.uuid4()` or equivalent — arbitrary strings are rejected.
- **Session persistence verified**: sessions are stored as `~/.claude/projects/<cwd-mangled>/<session-uuid>.jsonl` (standard disk files). Confirmed present across multiple days / machine state changes on this machine (oldest session observed: 2025-12-10). Resume survives restarts by virtue of filesystem persistence.

**Codex adapter:**
- Start: `codex exec --json "<prompt>"` (NOT `--output-format json` — that flag does not exist on codex exec)
- Session ID discovery: **verified** — `--json` emits JSONL with the first event being `{"type":"thread.started","thread_id":"<uuid>"}`. Parse that line; fall back to `session id:` header in plain-text mode.
- Resume: `codex exec resume <thread_id> "<prompt>"`
- Sample first-turn JSONL (captured 2026-04-21, codex v0.121):
  ```
  {"type":"thread.started","thread_id":"019db172-4d01-7072-860c-99114afe8792"}
  {"type":"turn.started"}
  {"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"OK"}}
  {"type":"turn.completed","usage":{"input_tokens":27138,...}}
  ```

**Droid adapter:**
- Start: `droid exec --auto low --output-format json "<prompt>"`
- Session ID discovery: **verified** — `--output-format json` emits a single JSON object on stdout with `session_id` as a first-class field. Filesystem fallback (`~/.factory/sessions/<cwd-mangled>/last.settings.json`) remains available if `--output-format json` is unavailable in future versions.
- Resume: `droid exec --auto low -s <session_id> "<prompt>"`
- Sample first-turn JSON (captured 2026-04-21, droid v0.103):
  ```json
  {"type":"result","subtype":"success","is_error":false,"duration_ms":4637,
   "num_turns":1,"result":"OK","session_id":"0cd2d3a0-81ef-47d1-9216-9beaeba60455",
   "usage":{"input_tokens":6,"output_tokens":6,...}}
  ```

### Persistence layout (repo-local, not home-scoped)

```
.aragora/agent_bridge/runs/<run_id>/
├── run.json              # metadata: task, created_at, agents list, status
├── sessions.json         # {role: {harness, session_id, worktree, branch}}
├── events.jsonl          # append-only turn log with timestamps
└── turns/
    ├── 001-codex-reviewer.md
    ├── 002-droid-synthesizer.md
    └── ...
```

Rationale for repo-local over `~/.aragora/`: enables worktree-scoped testing, survives machine migration, reviewable in PRs. Home-scoped paths remain only for legacy compat with existing `agent_bridge.py`.

### Footer contract

Every brokered turn must end with:

```
---BRIDGE-FOOTER---
summary: <one-sentence>
next_actor: <role or null>
needs_human: <bool>
done: <bool>
artifacts: [<paths>]
tests_run: [<commands>]
```

If the footer is missing or malformed, broker:
1. Records the raw turn in `events.jsonl` with `parse_status: malformed`
2. Dispatches a repair prompt to the same harness/session
3. Only advances the baton when a valid footer is received

Repair budget: 1 retry. After that, the turn is surfaced for human review.

### Worktree reuse

Each logical agent gets its own worktree via the existing `scripts/codex_worktree_autopilot.py` before the first turn. Broker stores `worktree_path` and `branch` alongside the session ID in `sessions.json`. This keeps agent file edits isolated and enables `git worktree remove` on run completion.

**Impedance note for PR 2 implementers**: `codex_worktree_autopilot.py` exposes `--agent <name>` (defaults to `codex`) and is keyed on the *harness track*, not on the *logical role* (reviewer / synthesizer / etc.). The broker must invoke it N times with distinct slugs per role (e.g., `--agent bridge-reviewer-a`, `--agent bridge-reviewer-b`), or add a thin wrapper. This is NOT a drop-in call.

## Delivery sequence

| PR | Scope | Est. size |
|---|---|---|
| **PR 1 (this doc)** | Design refresh, transport proof, test fixtures | docs-only, this PR |
| **PR 2** | Broker core: session registry, per-harness adapters, run/event persistence, CLI entrypoint `scripts/agent_bridge_broker.py`, single-run scripted turn dispatch | ~2 days |
| **PR 3** | Observable UI: read APIs, Autonomous bridge list/detail pages under `aragora/live/src/app/(app)/autonomous/bridge/`, unified activity feed (polling v0) | ~1-2 days |
| **PR 4 (optional)** | Supervisory features: pause/resume/cancel, decision cards, policy gates, many-model escalation, PR publication boundary | weeks |

PR 4 scope only dispatched if PRs 1–3 prove insufficient.

## v0 scope boundary

Explicitly **in scope** for PR 2 (broker core):
- Persistent context per harness via resume APIs
- Manual/scripted turn dispatch from CLI
- Run/event persistence
- Worktree isolation per agent
- Footer-contract enforcement

Explicitly **out of scope** for v0:
- No tmux keystroke driving (legacy tools keep working, separately)
- No autonomous PR merge
- No PR publication
- No policy engine
- No many-model debate escalation
- No websocket streaming (polling in v0)

## Relationship to existing work

- `scripts/agent_bridge.py` — keep. Legacy manual-operator surface for tmux-backed sessions. Broker does **not** retrofit it.
- `aragora/swarm/session_mux.py` — keep. Legacy session muxing.
- `aragora/harnesses/` (base, claude_code, codex, adapter) — refactor target. The new broker's per-harness adapters live under `aragora/swarm/agent_bridge/harnesses/` and may import/reuse code from the existing `aragora/harnesses/` package. No duplication; existing classes are reused where the session-mode matches.
- `~/.aragora/agent-bridge/sessions.json` — existing file. Broker reads it for legacy session inventory; new runs write to repo-local `.aragora/agent_bridge/runs/`.
- `.aragora/codex-reports/` — existing directory (adopted today). Broker optionally symlinks per-run event logs here for Claude Code consumption until PR 3 ships the UI.

## Test plan

### PR 1 (this doc): no code, fixture harvesting

- Include the four-turn cross-harness test transcript above as reproducible evidence.
- Capture sample Codex `exec --json` output for one turn (manual pre-flight; codex exec uses `--json`, not `--output-format`).
- Capture sample Droid `exec --output-format json` output for one turn (manual pre-flight).
- Fixtures land under `tests/fixtures/agent_bridge/` to be consumed by PR 2's adapter tests.

### PR 2: broker core

- Unit tests per adapter (Codex, Droid, Claude): parse session ID from start-turn output; verify resume preserves session ID.
- Unit tests for footer extraction + malformed-footer repair.
- Unit tests for event-log append semantics (idempotency under broker restart).
- Integration test with mocked subprocesses: start → multiple resumes → restart broker → reload run state → continue.
- One gated **live** smoke test (opt-in via env var, not in CI default) that reproduces the four-turn cross-harness test end-to-end via the broker.

### PR 3: UI

- API tests: run list, run detail, event stream.
- UI tests: run list renders, run detail shows interleaved turns per agent, pending-human gates render distinctly.

## Resolved spikes (2026-04-21)

Questions 1–3 from the original draft are resolved inline via the per-harness adapter sections above. Briefly:

1. **Codex `exec --json`**: emits `{"type":"thread.started","thread_id":"<uuid>"}` as the first JSONL event. First-class field. Parse this; fall back to `session id:` header.
2. **Droid `exec --output-format json`**: emits a single JSON object with `session_id` as a first-class field. First-class. Filesystem fallback still available.
3. **Claude `--session-id` survival**: sessions persist as `~/.claude/projects/<cwd-mangled>/<session-uuid>.jsonl`. Disk-backed, survives restarts by filesystem persistence (oldest session observed on primary dev machine: 2025-12-10).

## Remaining open questions (for PR 2 author)

1. Footer contract: should the broker inject the footer template into every outgoing prompt, or rely on each agent to know the contract via their system prompt? (Leaning inject — less surprise, more reliable on fresh sessions.)
2. Worktree cleanup: automatic on run completion, or operator-triggered? (Leaning operator-triggered for v0 — safer against partial-run salvage.)

## Amendments applied from Codex's original plan

Three amendments to the plan posted in conversation 2026-04-21:

1. **Session-ID discovery** (Codex and Droid): plan assumed JSON-mode parsing. Validation used plain-text header/filesystem fallbacks. PR 2 should prefer JSON if clean, fall back to the validated alternatives.
2. **Empirical test evidence**: the four-turn cross-harness transcript above is preserved as appendix evidence. Future reviewers see actual validated outputs.
3. **Claude Code session ID**: broker-assigned UUID (confirmed per `claude --help` showing `--session-id`), not auto-captured from CLI output.

## Non-goals

- Not a replacement for Codex cloud session orchestration. The cloud sessions at factory.ai / chat.openai.com remain paste-based unless the respective vendors expose APIs.
- Not a multi-tenant production service. Single-operator, single-repo, local-first v1.
- Not an auto-merge pipeline. Human settlement remains mandatory per the `#6279` design.

## Appendix A — empirical command ledger (2026-04-21)

Full commands, session IDs, and outputs from the validation test are captured verbatim in the "Empirical validation" section above. Reproducible on any machine with `codex` (OpenAI CLI ≥ 0.121) and `droid` (Factory ≥ 0.103) installed and authenticated.
