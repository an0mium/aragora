# Agent Bridge PR 2 Scoping

**Status:** Canonical contract for PR 2a redo. Authored 2026-04-21 via Codex scoping pass; committed as source of truth after PR #6387 was closed BLOCKED on joint Codex + Droid verdict (foundation-level deviations from this scope — footer format, persistence schema, identity model, harness layering, tri-layer coupling).

**Predecessor design:** `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md` (lines 64-79, 85-159, 160-211, 226-229 are load-bearing for this scope).

**PR split:** PR 2a implements §1-§5 (backend core) behind this contract. PR 2b layers read API. PR 2c layers autonomous/bridge UI.

Scope lock:
- PR 2a is backend-only plus `scripts/agent_bridge_broker.py`.
- No UI, no polling/websocket work, no pause/resume/cancel, no policy gate, no PR publication, no edits to `scripts/agent_bridge.py`, and no edits to `aragora/swarm/session_mux.py`.

## 1. Exact File Layout

Reference: `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:64-72,85-159,190-195`

New package files under `aragora/swarm/agent_bridge/`:

| File | Purpose |
| --- | --- |
| `aragora/swarm/agent_bridge/__init__.py` | Public exports for broker, models, persistence helpers, and transport protocol. |
| `aragora/swarm/agent_bridge/models.py` | Dataclasses and enums for run state, session registry entries, footer payloads, parsed turns, and event records. |
| `aragora/swarm/agent_bridge/errors.py` | Typed exception hierarchy for transport, persistence, footer, and broker state failures. |
| `aragora/swarm/agent_bridge/transport.py` | `Transport` protocol plus shared subprocess result dataclasses (`SessionStart`, `TurnResult`, `ParsedTurn`, `HealthStatus`). |
| `aragora/swarm/agent_bridge/footer.py` | Footer template injection, footer extraction, validation, and repair-prompt construction. |
| `aragora/swarm/agent_bridge/persistence.py` | Repo-local path helpers, atomic JSON writes, JSONL append/read helpers, transcript rendering, and run/session reload. |
| `aragora/swarm/agent_bridge/worktree.py` | Thin wrapper around `scripts/codex_worktree_autopilot.py ensure` plus branch/path discovery per role. |
| `aragora/swarm/agent_bridge/broker.py` | `start_run`, `dispatch_turn`, `show_run`, and `list_runs` orchestration. |
| `aragora/swarm/agent_bridge/harnesses/__init__.py` | Harness adapter registry and factory. |
| `aragora/swarm/agent_bridge/harnesses/claude.py` | Claude Code CLI-resume transport adapter. |
| `aragora/swarm/agent_bridge/harnesses/codex.py` | Codex CLI-resume transport adapter. |
| `aragora/swarm/agent_bridge/harnesses/droid.py` | Droid CLI-resume transport adapter. |

Supporting non-package files:

| File | Purpose |
| --- | --- |
| `scripts/agent_bridge_broker.py` | Thin CLI entrypoint that parses args, calls broker functions, prints text/JSON, and returns stable exit codes. |

New tests and fixtures:

| File | Purpose |
| --- | --- |
| `tests/swarm/test_agent_bridge_footer.py` | Footer extraction and repair behavior. |
| `tests/swarm/test_agent_bridge_persistence.py` | `run.json`, `sessions.json`, `events.jsonl`, and transcript roundtrip behavior. |
| `tests/swarm/test_agent_bridge_transport_claude.py` | Claude adapter start/resume parsing and error handling. |
| `tests/swarm/test_agent_bridge_transport_codex.py` | Codex adapter JSONL parsing, header fallback, and resume behavior. |
| `tests/swarm/test_agent_bridge_transport_droid.py` | Droid adapter JSON parsing and filesystem fallback behavior. |
| `tests/swarm/test_agent_bridge_broker.py` | End-to-end broker state transitions with mocked worktree + transport layers. |
| `tests/swarm/test_agent_bridge_live_smoke.py` | Opt-in live smoke test for the four-turn cross-harness flow. |
| `tests/scripts/test_agent_bridge_broker.py` | CLI parser, exit code, and JSON/text output coverage. |
| `tests/fixtures/agent_bridge/codex_start.jsonl` | Captured `codex exec --json` first-turn fixture. |
| `tests/fixtures/agent_bridge/codex_resume.jsonl` | Captured `codex exec resume --json` fixture. |
| `tests/fixtures/agent_bridge/droid_start.json` | Captured `droid exec --output-format json` first-turn fixture. |
| `tests/fixtures/agent_bridge/droid_resume.json` | Captured `droid exec -s ... --output-format json` fixture. |
| `tests/fixtures/agent_bridge/claude_start.txt` | Captured `claude -p --session-id ...` text fixture. |
| `tests/fixtures/agent_bridge/claude_resume.txt` | Captured `claude -p --resume ...` text fixture. |

## 2. Transport Interface Signature

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:73-84`

Concrete module: `aragora/swarm/agent_bridge/transport.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .models import BridgeFooter

ParseStatus = Literal["ok", "missing", "malformed"]

@dataclass(frozen=True)
class HealthStatus:
    harness: str
    ok: bool
    version: str | None
    binary_path: str | None
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ParsedTurn:
    session_id: str | None
    message_text: str
    footer_raw: str | None
    footer: BridgeFooter | None
    parse_status: ParseStatus
    parse_errors: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    provider_payload: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SessionStart:
    harness: str
    session_id: str
    cwd: Path
    model: str
    command: list[str]
    exit_code: int
    raw_stdout: str
    raw_stderr: str
    parsed: ParsedTurn

@dataclass(frozen=True)
class TurnResult:
    harness: str
    session_id: str
    command: list[str]
    exit_code: int
    raw_stdout: str
    raw_stderr: str
    parsed: ParsedTurn

class Transport(Protocol):
    harness: str

    def start_session(self, prompt: str, *, cwd: Path, model: str) -> SessionStart: ...
    def resume_turn(self, session_id: str, prompt: str) -> TurnResult: ...
    def parse_response(self, raw: str) -> ParsedTurn: ...
    def healthcheck(self) -> HealthStatus: ...
```

Exception semantics:
- `healthcheck()` returns `HealthStatus(ok=False, ...)`; it does not raise for a missing binary.
- `start_session()` raises `TransportNotAvailableError` if `healthcheck().ok` is false before subprocess execution.
- `start_session()` raises `TransportLaunchError` when the subprocess exits nonzero or exits zero but no session ID can be established.
- `resume_turn()` raises `TransportResumeError` when the subprocess exits nonzero.
- `parse_response()` raises `TransportOutputParseError` only for harness-shape failures such as invalid JSON, missing `thread.started` on start, or missing `result`/agent message content.
- Malformed or missing bridge footer is not a transport exception. It returns `ParsedTurn(parse_status="missing" | "malformed", footer=None, parse_errors=[...])` so the broker can issue a repair prompt.

Important construction rule:
- The protocol intentionally omits `cwd` on `resume_turn()` because the broker will build a per-role adapter instance from `sessions.json` before each dispatch. The adapter instance owns immutable `cwd` and `model` for that role.

## 3. Per-Harness Adapter Implementations

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:85-117`
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:220-237`

Common subprocess rules for all three adapters:
- Use `subprocess.run([...], cwd=<worktree_path>, text=True, capture_output=True, check=False)`.
- Never use `shell=True`.
- Record `command`, `returncode`, `stdout`, and `stderr` verbatim into the returned `SessionStart` or `TurnResult`.
- Treat non-empty `stderr` as diagnostic only when `returncode == 0`; do not fail on warnings alone.

### Claude Code adapter

Module:
- `aragora/swarm/agent_bridge/harnesses/claude.py`

Start invocation:

```python
[
    "claude",
    "-p",
    "--session-id",
    str(broker_uuid),
    "--model",
    model,
    prompt,
]
```

Resume invocation:

```python
[
    "claude",
    "-p",
    "--resume",
    session_id,
    "--model",
    model,
    prompt,
]
```

Session ID source:
- Broker-assigned UUID is canonical.
- The adapter must generate it before the start call with `uuid.uuid4()` and validate that it is a real UUID string.
- No stdout parsing for session ID is needed.

`parse_response(raw)` behavior:
- `message_text = raw.strip()`
- `footer_raw` and `footer` come from `footer.extract_footer(...)`
- `session_id` stays `None` in `ParsedTurn`; the caller already knows it from the start/resume context

Error surfaces:
- Invalid UUID rejected by Claude CLI: `TransportLaunchError` with `stderr`.
- Exit code nonzero: `TransportLaunchError` or `TransportResumeError`.
- Zero exit with empty stdout: `TransportOutputParseError`.

Persistence note:
- Session persistence is disk-backed at `~/.claude/projects/<cwd-mangled>/<session-uuid>.jsonl`, per the design doc. PR 2 only needs that fact for pre-flight validation, not for runtime parsing.

### Codex adapter

Module:
- `aragora/swarm/agent_bridge/harnesses/codex.py`

Start invocation:

```python
[
    "codex",
    "exec",
    "--json",
    "--model",
    model,
    prompt,
]
```

Resume invocation:

```python
[
    "codex",
    "exec",
    "resume",
    "--json",
    "--model",
    model,
    session_id,
    prompt,
]
```

Session ID source:
- Primary: parse the first JSONL event with `type == "thread.started"` and read `thread_id`.
- Fallback for start only: if JSON decoding fails but exit code is zero, scan plain stdout for `session id: <uuid>` and use that.
- Resume never discovers a new session ID; it reuses the stored one. If a resume JSONL stream unexpectedly emits `thread.started`, it must match the stored session ID or the adapter raises `TransportOutputParseError`.

`parse_response(raw)` behavior:
- Parse stdout as JSONL, one object per line.
- Concatenate every `item.completed` where `item.type == "agent_message"` and `item.text` is present, in stream order, separated by `\n\n`.
- Pull usage from the final `turn.completed.usage` object if present.
- Pass the concatenated assistant text through footer extraction.

Error surfaces:
- Exit code nonzero: `TransportLaunchError` or `TransportResumeError`.
- Start output missing `thread.started`: `TransportOutputParseError`.
- JSONL present but no `agent_message`: `TransportOutputParseError`.
- Plain-text fallback path missing both JSON and `session id:` header: `TransportOutputParseError`.

### Droid adapter

Module:
- `aragora/swarm/agent_bridge/harnesses/droid.py`

Start invocation:

```python
[
    "droid",
    "exec",
    "--auto",
    "low",
    "--output-format",
    "json",
    "--model",
    model,
    "--cwd",
    str(cwd),
    prompt,
]
```

Resume invocation:

```python
[
    "droid",
    "exec",
    "--auto",
    "low",
    "--output-format",
    "json",
    "-s",
    session_id,
    "--model",
    model,
    "--cwd",
    str(cwd),
    prompt,
]
```

Session ID source:
- Primary: parse stdout as a single JSON object and read `session_id`.
- Filesystem fallback: if JSON mode succeeds but the object lacks `session_id`, inspect `~/.factory/sessions/<cwd-mangled>/`, ignore `last.*`, and take the basename of the newest `*.settings.json` file as the session ID.

`parse_response(raw)` behavior:
- Parse stdout as a single JSON object.
- `message_text = payload["result"]`
- `usage = payload.get("usage", {})`
- Pass `message_text` through footer extraction.

Error surfaces:
- Exit code nonzero: `TransportLaunchError` or `TransportResumeError`.
- Zero exit with invalid JSON: `TransportOutputParseError`.
- JSON present but missing `result`: `TransportOutputParseError`.
- Fallback directory missing or containing no concrete `*.settings.json` files: `TransportOutputParseError`.

## 4. Persistence Schema

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:118-131`
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:147-152`

Root layout:

```text
.aragora/agent_bridge/runs/<run_id>/
├── run.json
├── sessions.json
├── events.jsonl
└── turns/
    ├── 001-codex-implementer.md
    ├── 002-claude-reviewer.md
    └── ...
```

Write semantics:
- `run.json` and `sessions.json` are rewritten atomically via `*.tmp` + `Path.replace()`.
- `events.jsonl` is append-only.
- Every JSONL event includes deterministic `event_id`; append is a no-op if that `event_id` already exists. That is the restart idempotency contract.

### `run.json`

Exact on-disk shape:

```json
{
  "schema_version": 1,
  "run_id": "bridge_20260421T191953Z_pr6306",
  "task": "Review and refine the protocol orchestrator implementation plan.",
  "status": "running",
  "created_at": "2026-04-21T19:19:53Z",
  "updated_at": "2026-04-21T19:24:10Z",
  "completed_at": null,
  "last_turn_index": 1,
  "next_actor": "reviewer",
  "repair_budget_per_turn": 1,
  "footer_mode": "prompt_injected",
  "worktree_cleanup_mode": "operator_triggered",
  "participants": [
    {
      "role": "implementer",
      "harness": "codex",
      "model": "gpt-5.4"
    },
    {
      "role": "reviewer",
      "harness": "claude_code",
      "model": "claude-opus-4-7"
    }
  ]
}
```

### `sessions.json`

Exact on-disk shape:

```json
{
  "schema_version": 1,
  "run_id": "bridge_20260421T191953Z_pr6306",
  "updated_at": "2026-04-21T19:24:10Z",
  "sessions": {
    "implementer": {
      "role": "implementer",
      "harness": "codex",
      "model": "gpt-5.4",
      "session_id": "019db172-4d01-7072-860c-99114afe8792",
      "worktree_agent_slug": "bridge-pr6306-implementer",
      "worktree_path": "/Users/armand/Development/aragora/.worktrees/codex-auto/bridge-pr6306-implementer",
      "branch": "codex/bridge-pr6306-implementer",
      "session_status": "active",
      "started_at": "2026-04-21T19:21:02Z",
      "last_turn_index": 1,
      "last_completed_at": "2026-04-21T19:24:10Z"
    },
    "reviewer": {
      "role": "reviewer",
      "harness": "claude_code",
      "model": "claude-opus-4-7",
      "session_id": null,
      "worktree_agent_slug": "bridge-pr6306-reviewer",
      "worktree_path": "/Users/armand/Development/aragora/.worktrees/codex-auto/bridge-pr6306-reviewer",
      "branch": "codex/bridge-pr6306-reviewer",
      "session_status": "not_started",
      "started_at": null,
      "last_turn_index": 0,
      "last_completed_at": null
    }
  }
}
```

### `events.jsonl`

One JSON object per line. Exact example `turn.result` record:

```json
{
  "schema_version": 1,
  "event_id": "bridge_20260421T191953Z_pr6306:turn:001:result:0",
  "run_id": "bridge_20260421T191953Z_pr6306",
  "ts": "2026-04-21T19:24:09Z",
  "event_type": "turn.result",
  "turn_index": 1,
  "role": "implementer",
  "harness": "codex",
  "session_id": "019db172-4d01-7072-860c-99114afe8792",
  "parse_status": "ok",
  "payload": {
    "exit_code": 0,
    "command": [
      "codex",
      "exec",
      "resume",
      "--json",
      "--model",
      "gpt-5.4",
      "019db172-4d01-7072-860c-99114afe8792",
      "Review the plan and emit the footer."
    ],
    "transcript_path": ".aragora/agent_bridge/runs/bridge_20260421T191953Z_pr6306/turns/001-codex-implementer.md",
    "footer": {
      "summary": "Outlined the persistence and transport slices.",
      "next_actor": "reviewer",
      "needs_human": false,
      "done": false,
      "artifacts": [],
      "tests_run": []
    }
  }
}
```

### Per-turn transcript files

Contract:
- Markdown with YAML front matter.
- Filename format: `NNN-<harness>-<role>.md`.
- Front matter keys: `schema_version`, `run_id`, `turn_index`, `role`, `harness`, `model`, `session_id`, `started_at`, `completed_at`, `exit_code`, `parse_status`, `repair_attempts`.
- Body sections in order: `## Prompt`, `## Raw Stdout`, `## Raw Stderr`, `## Parsed Message`, `## Footer`.
- Repair attempts append `## Repair Attempt 1`, `## Repair Attempt 2`, and so on inside the same file.

Example front matter and section order:

```yaml
---
schema_version: 1
run_id: bridge_20260421T191953Z_pr6306
turn_index: 1
role: implementer
harness: codex
model: gpt-5.4
session_id: 019db172-4d01-7072-860c-99114afe8792
started_at: 2026-04-21T19:23:40Z
completed_at: 2026-04-21T19:24:09Z
exit_code: 0
parse_status: ok
repair_attempts: 0
---

## Prompt
...

## Raw Stdout
...

## Raw Stderr
...

## Parsed Message
...

## Footer
summary: Outlined the persistence and transport slices.
next_actor: reviewer
needs_human: false
done: false
artifacts: []
tests_run: []
```

## 5. Footer Contract Enforcement

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:133-152`
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:228`

Concrete implementation shape in `aragora/swarm/agent_bridge/footer.py`:
- `FOOTER_MARKER = "---BRIDGE-FOOTER---"`
- `build_footer_instruction(*, roles: list[str]) -> str`
- `extract_footer_block(text: str) -> str | None`
- `parse_footer_block(block: str, *, allowed_roles: set[str]) -> FooterParseResult`
- `build_repair_prompt(*, parse_errors: list[str], original_message: str, allowed_roles: set[str]) -> str`

Validation rules:
- The final non-whitespace block must start with `---BRIDGE-FOOTER---`.
- Required keys are exactly `summary`, `next_actor`, `needs_human`, `done`, `artifacts`, `tests_run`.
- `summary` must be a single non-empty line.
- `next_actor` must be one of the run’s participant roles or `null`.
- `needs_human` and `done` must be lowercase `true` or `false`.
- `artifacts` and `tests_run` must be JSON arrays of strings, even when empty.
- Unknown keys are rejected in v0.

Valid footer example:

```text
---BRIDGE-FOOTER---
summary: Added run persistence and turn transcript rendering.
next_actor: reviewer
needs_human: false
done: false
artifacts: ["aragora/swarm/agent_bridge/persistence.py"]
tests_run: ["pytest tests/swarm/test_agent_bridge_persistence.py -q"]
```

Malformed footer example the broker must detect:

```text
---BRIDGE-FOOTER---
summary Added run persistence.
next_actor: qa
needs_human: no
done: false
artifacts: aragora/swarm/agent_bridge/persistence.py
tests_run: pytest tests/swarm/test_agent_bridge_persistence.py -q
```

Why the malformed example fails:
- `summary` is missing the colon delimiter.
- `next_actor` is not a known role and is not `null`.
- `needs_human` is not `true|false`.
- `artifacts` is not a JSON array.
- `tests_run` is not a JSON array.

Repair flow:
1. Broker records a `turn.result` event with `parse_status: "missing"` or `"malformed"`.
2. Broker writes the raw transcript file before repair so the failure is inspectable.
3. Broker dispatches one repair prompt to the same harness session. This is the same turn index with `repair_attempts = 1`.
4. The repair prompt asks for footer-only output, not a fresh full answer.
5. If the repair output validates, broker appends `turn.repair_requested` and `turn.completed`, rewrites the transcript file with a `## Repair Attempt 1` section, and advances the baton.
6. If the repair output still fails, broker sets `run.status = "awaiting_human"` and returns exit code `4`.

Repair prompt shape:

```text
Your previous response did not satisfy the bridge footer contract.
Return ONLY a corrected footer block that starts with ---BRIDGE-FOOTER---.
Do not restate your full answer.

Validation errors:
- <error 1>
- <error 2>

Allowed next_actor values: <comma-separated roles>, null
Required fields: summary, next_actor, needs_human, done, artifacts, tests_run
```

## 6. CLI Entrypoint

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:165-186`

File:
- `scripts/agent_bridge_broker.py`

Global exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Bad CLI usage or mutually exclusive/invalid args |
| `2` | Run, role, or state not found / illegal transition |
| `3` | Transport or worktree subprocess failure |
| `4` | Footer still missing or malformed after the one repair retry |
| `5` | Persistence read/write failure |
| `6` | Pre-flight healthcheck failed for a requested harness |

### `start-run`

Purpose:
- Create the run directory.
- Provision one isolated worktree per role.
- Write `run.json`, `sessions.json`, and initial events.
- Do not start harness sessions yet.

Args:

```text
python3 scripts/agent_bridge_broker.py start-run \
  --task "..." | --task-file /abs/path/task.md \
  --actor <role:harness:model> \
  [--actor <role:harness:model> ...] \
  [--run-id <slug>] \
  [--base main] \
  [--json]
```

Rules:
- `--actor` is repeatable and required at least once.
- `harness` values for v0 are `claude_code`, `codex`, `droid`.
- If `--run-id` is omitted, broker generates `bridge_<UTC timestamp>`.
- Healthchecks run for all distinct harnesses before worktree provisioning.

### `dispatch-turn`

Purpose:
- If the role has no `session_id`, call `start_session(...)`.
- Otherwise call `resume_turn(...)`.
- Enforce footer contract.
- Write turn transcript and events.
- Update `run.json` and `sessions.json`.

Args:

```text
python3 scripts/agent_bridge_broker.py dispatch-turn \
  --run-id <id> \
  --role <role> \
  --prompt "..." | --prompt-file /abs/path/prompt.md \
  [--json]
```

Rules:
- `dispatch-turn` is explicit; no autonomous baton scheduler in PR 2.
- The broker trusts `--role`; it does not auto-dispatch to `next_actor`.
- If the turn footer sets `done: true`, the broker marks the run `completed`.
- If the turn footer sets `needs_human: true`, the broker marks the run `awaiting_human`.

### `show-run`

Purpose:
- Print a compact summary of run metadata, participants, latest status, and latest turn.

Args:

```text
python3 scripts/agent_bridge_broker.py show-run --run-id <id> [--json]
```

### `list-runs`

Purpose:
- Enumerate `.aragora/agent_bridge/runs/*/run.json` and print one line or one JSON object per run.

Args:

```text
python3 scripts/agent_bridge_broker.py list-runs [--status running|awaiting_human|completed|failed] [--json]
```

Out of scope for v0:
- `pause-run`
- `resume-run`
- `cancel-run`
- any automatic scheduler that loops on `next_actor`

## 7. Test Matrix

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:196-211`

| Surface | Files | Mock approach | Example test case |
| --- | --- | --- | --- |
| Adapter parsing | `tests/swarm/test_agent_bridge_transport_codex.py`, `tests/swarm/test_agent_bridge_transport_droid.py`, `tests/swarm/test_agent_bridge_transport_claude.py`, fixtures under `tests/fixtures/agent_bridge/` | Monkeypatch `subprocess.run` to return captured stdout/stderr fixtures; no real CLI calls in unit tests | Codex start fixture begins with `{"type":"thread.started","thread_id":"019db172-4d01-7072-860c-99114afe8792"}` and the adapter returns that exact session ID plus assistant text `"OK"` |
| Footer extraction and repair | `tests/swarm/test_agent_bridge_footer.py` | Pure unit tests; no subprocesses; call parser and repair-prompt builder directly | The malformed footer example above returns `parse_status == "malformed"` and a repair prompt whose first instruction is `Return ONLY a corrected footer block` |
| Persistence roundtrip | `tests/swarm/test_agent_bridge_persistence.py` | Real `tmp_path` filesystem, deterministic clock/UUID monkeypatches, no subprocesses | Write `run.json`, `sessions.json`, append `turn.started` + `turn.result`, reload state, append the same `event_id` again, and assert the file still contains one copy |
| Single-run integration | `tests/swarm/test_agent_bridge_broker.py`, `tests/scripts/test_agent_bridge_broker.py` | Mock worktree wrapper and adapter factory; use real persistence on `tmp_path` | `start-run` provisions two roles, `dispatch-turn` starts Codex, broker reloads from disk, `dispatch-turn` resumes Claude, and `next_actor` advances from `reviewer` to `implementer` |

Opt-in live smoke:
- File: `tests/swarm/test_agent_bridge_live_smoke.py`
- Gate: `ARAGORA_LIVE_AGENT_BRIDGE=1`
- Behavior: reproduce the four-turn Codex ↔ Droid proof from the design doc against real installed CLIs and real credentials, but never run in default CI.

## 8. Defaults For The Two Remaining Open Questions

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:226-229`

Footer contract default:
- Inject the footer template into every outgoing prompt.
- Rationale: PR 2 is crossing three heterogeneous harnesses with different default system prompts; broker-side injection is the only v0 path that makes first-turn behavior deterministic and restart-safe.

Worktree cleanup default:
- Operator-triggered cleanup only.
- Rationale: the repo’s own worktree guidance biases toward reversibility and salvage. Automatic cleanup on `done: true` is too risky for v0 because a “completed” run can still need local inspection, transcript review, or manual publication follow-up.

## 9. Implementation Order

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:165-166`

1. Slice 1: package scaffolding, models, errors, persistence module, and persistence tests. This establishes the on-disk contract before any subprocess logic exists.
2. Slice 2: footer parser, footer instruction injection, repair-prompt builder, and footer tests. This isolates the strictest behavioral contract early.
3. Slice 3: transport protocol, Codex adapter, Codex fixtures, and Codex transport tests. Codex has the cleanest validated JSONL story and sets the parser pattern.
4. Slice 4: Claude adapter, Droid adapter, their fixtures, and adapter tests. This completes the three-harness matrix without touching broker orchestration yet.
5. Slice 5: worktree wrapper plus broker orchestration for `start_run`, `dispatch_turn`, `show_run`, and `list_runs`, with broker integration tests on real temp persistence.
6. Slice 6: `scripts/agent_bridge_broker.py` plus script-level CLI tests and the opt-in live smoke test skeleton.

## 10. Pre-Flight Checks

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:38-56`
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:95-116`

Run these before starting implementation.

Codex:

```bash
codex --version
codex exec --json "Respond with ONLY OK"
```

Verify:
- the CLI exists and authenticates
- stdout is JSONL
- the first line contains `{"type":"thread.started","thread_id":"..."}`

Then resume the returned thread ID:

```bash
codex exec resume --json <thread_id> "Respond with ONLY STILL_OK"
```

Verify:
- resume succeeds in the same repo
- stdout still uses JSONL
- the final `item.completed` message text is `STILL_OK`

Droid:

```bash
droid --version
droid exec --auto low --output-format json "Respond with ONLY OK"
```

Verify:
- stdout is a single JSON object
- it contains top-level `session_id`
- it contains top-level `result`

Then resume the returned session ID:

```bash
droid exec --auto low --output-format json -s <session_id> "Respond with ONLY STILL_OK"
```

Verify:
- resume succeeds
- stdout still contains the same `session_id`
- `result` is `STILL_OK`

Claude Code:

```bash
claude --version
UUID="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
claude -p --session-id "$UUID" "Respond with ONLY OK"
find ~/.claude/projects -name "${UUID}.jsonl"
claude -p --resume "$UUID" "Respond with ONLY STILL_OK"
```

Verify:
- non-interactive `-p` mode works
- the broker-assigned UUID is accepted
- a transcript file exists under `~/.claude/projects/.../${UUID}.jsonl`
- resume returns `STILL_OK`

Worktree autopilot:

```bash
python3 scripts/codex_worktree_autopilot.py ensure --agent bridge-preflight --base main --force-new --print-path
```

Verify:
- a disposable managed worktree is created
- `git -C <printed-path> rev-parse --abbrev-ref HEAD` returns a branch

## 11. Risks And Open-Ended Decisions

Reference:
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:107-110`
- `docs/plans/2026-04-21-agent-bridge-cli-resume-transport.md:198-203`

1. Droid filesystem fallback is not fully resolved by the design doc alone. The doc cites `~/.factory/sessions/<cwd-mangled>/last.settings.json`, but on this machine `last.settings.json` does not itself contain `session_id`; the concrete session ID is recoverable from the newest non-`last.*` `*.settings.json` filename. If the PR 2 author does not want to rely on that local observation, they need to decide whether v0 will ship with JSON-only Droid parsing and no filesystem fallback.
2. The design doc says PR 1 harvested adapter fixtures under `tests/fixtures/agent_bridge/`, but those files are not present in this worktree snapshot. Before coding begins, the author needs to decide whether PR 2 owns fixture creation or whether they must first sync to a main revision where those fixtures already exist.
