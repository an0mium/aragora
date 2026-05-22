# Agent Dispatch Reach Plan

**Status:** plan-only (P54). No code yet. Five phased PRs scoped below; numbers
are tentative and may shift to dodge codex-namespace contention at claim time.

**Scope:** four capabilities the current aragora dispatch toolkit cannot do
today:

1. **Reach INTO a Droid CLI session** running in a Factory.ai web-CLI or local terminal that aragora did NOT launch.
2. **Reach INTO a Codex Desktop tab** running in the macOS Codex.app, whose IPC is private to the Electron process.
3. **Direct programmatic dispatch to "all currently-active agents"** including ones not registered via `agent_bridge.py launch`.
4. **One-button wake-up** for any agent — operator types one CLI command and the right agent gets the right prompt via the right backend.

The current dispatch surface (tmux launcher, send-prompt, multi-agent dialog, swarm supervisor, steering mailbox) handles **anything aragora itself launched** plus **delayed async** for anything else. This plan closes the gap to **synchronous reach** for the four cases above.

## Threat model + design constraints

- **Trust:** the operator is trusted. We are not designing against a malicious-operator attacker. The threat is *uncoordinated agents stepping on each other* + *operator can't reach the agent that's stuck*.
- **Determinism:** all delivery paths must produce an auditable receipt. No "fire and pray."
- **Idempotency:** redelivery of the same message must be detectable + deduplicable on the recipient side.
- **Backward-compatibility:** the v1.0 mailbox schema (Phase B) is frozen. New backends MUST consume it as-is; only the *transport* changes per backend.
- **Pure stdlib + OS-level scripting only.** No new pip deps. No third-party automation frameworks (Playwright comes in only at Phase 5 and only as an OPT-IN extra).

## Architecture — one unified dispatch surface

The current state has many specialized dispatchers (`tmux_send_prompt.sh`, `agent_bridge.py send`, `send_operator_steering.py`, etc.). The reach plan unifies them behind one CLI:

```
scripts/wake_agent.sh --lane LANE [--prompt TEXT | --prompt-file PATH] [--priority {low,normal,high,blocking}]
```

`wake_agent.sh` does:

1. Look up the lane's owner via `scripts/identify_lane_owner.py --json` (Phase A — already shipped).
2. Determine the **contact method** for that owner (see Phase 2 below — new `contact_method` field on `LaneRecord`).
3. Dispatch via the corresponding backend:
   - `tmux:<pane-name>` → `scripts/tmux_send_prompt.sh`
   - `osascript:codex-desktop:<thread-id>` → `scripts/codex_desktop_inject.sh` (Phase 3 — new)
   - `osascript:droid-cli:<window-title>` → `scripts/droid_cli_inject.sh` (Phase 4 — new)
   - `factory-api:<session-id>` → `scripts/factory_api_send.py` (Phase 4 — new, opt-in if Factory exposes API)
   - `mailbox-only:<owner_session>` → `scripts/send_operator_steering.py` (Phase B — already shipped)
4. On any path, write a delivery receipt to `.aragora/dispatch-receipts/<utc>-<lane>-<short-uuid>.json` with the chosen backend, the message SHA-256, and the dispatch outcome.

Backends fail gracefully: if `osascript:codex-desktop` fails, fall back to `mailbox-only` and emit a notification telling the operator to forward manually.

## Phase 1 — `contact_method` field on `LaneRecord` (smallest unblocker)

**New PR. Targets:** `scripts/agent_bridge.py` (extend `LaneRecord` dataclass) + `scripts/claim_active_agent_lane.py` (CLI flag).

```
# LaneRecord additive field
contact_method: str | None = None    # e.g. "tmux:claude-p52", "osascript:codex-desktop:019e...", "mailbox-only"
contact_payload: dict | None = None  # optional structured detail per backend
```

CLI flag (additive):

```
python3 scripts/claim_active_agent_lane.py ... \
    --contact-method tmux:claude-p52 \
    --contact-payload '{"pane": "claude-p52", "log": "~/.aragora/tmux-sessions/claude-p52.log"}'
```

Sessions launched via `agent_bridge.py launch` auto-populate `contact_method=tmux:<name>`. Sessions claimed by hand can fill it themselves. Sessions that can't (Codex Desktop, Droid web) get `contact_method=mailbox-only`.

Tests: ≥5 — happy path; default None; auto-populate from `agent_bridge.py launch`; CLI flag override; round-trip through identify_lane_owner.

Acceptance: every active lane in `.aragora/agent-bridge/lanes.json` has either a usable `contact_method` or explicit `mailbox-only`. Existing lanes get backfilled by a one-time sweeper script that infers method from `branch` / `worktree` patterns.

Bounded: ≤30 min · single PR · pure stdlib.

## Phase 2 — `scripts/wake_agent.sh` unified CLI (the "one button")

**New PR. Targets:** new file `scripts/wake_agent.sh` + new tests.

CLI:

```
wake_agent.sh --lane LANE                    # uses lane's contact_method
              [--prompt TEXT | --prompt-file PATH]
              [--priority {low,normal,high,blocking}]
              [--dry-run]                    # default: print intended action, don't dispatch
              [--apply]                      # opt-in mutate
              [--json]
              [--fallback {mailbox-only,fail}] # default: mailbox-only
```

Execution:

1. `identify_lane_owner.py --lane-id <LANE> --json` → owner + contact_method
2. Resolve backend:
   - `tmux:NAME` → spawn `tmux_send_prompt.sh --name NAME --prompt-file PATH`
   - `osascript:codex-desktop:THREAD` → `codex_desktop_inject.sh --thread THREAD --prompt-file PATH` (Phase 3)
   - `osascript:droid-cli:WINDOW` → `droid_cli_inject.sh --window WINDOW --prompt-file PATH` (Phase 4)
   - `factory-api:SESSION` → `factory_api_send.py --session SESSION --prompt-file PATH` (Phase 4 opt-in)
   - `mailbox-only:OWNER` → `send_operator_steering.py --to OWNER --body-file PATH`
   - missing/unknown → fall back to `--fallback`
3. Always write `.aragora/dispatch-receipts/<utc>-<lane>-<sha8>.json`
4. On `osascript:*` backend success, also write a `tail`-able log entry to `~/.aragora/dispatch-log.jsonl`

Tests: ≥8 — each backend path's dispatch + fallback + receipt write + dry-run + missing-lane + missing-contact-method + permission failure + osascript not available.

Bounded: ≤45 min · single PR · pure stdlib + bash.

## Phase 3 — `scripts/codex_desktop_inject.sh` (osascript bridge)

**The hardest piece.** Codex Desktop is an Electron app on macOS with no documented external IPC. Options investigated:

| Path | Verdict |
|---|---|
| HTTP API on localhost | Investigate first — if Codex Desktop exposes a local API (some Electron apps do via `localhost:<random-port>`), use it directly. Cleanest if available. |
| osascript / Accessibility API | Frontmost window automation: focus the Codex.app, find the right tab via window-title match, simulate keystrokes into the prompt input. Brittle but achievable. |
| Filesystem inbox Codex polls | Requires Codex Desktop to opt in to polling. No mechanism today. Out of scope. |
| URL scheme deep-link | `codex://thread/<id>` may exist for tab-open; doesn't send a prompt. Useful for context but not the primary path. |

**Implementation (osascript path):**

```bash
# scripts/codex_desktop_inject.sh --thread <id> --prompt-file <path>
# - Read prompt content from file (UTF-8)
# - Run osascript to:
#     activate Codex Desktop
#     find the window whose tab matches the thread-id (heuristic: tab title contains thread short-id)
#     set focus to the prompt input textarea
#     paste prompt content (via NSPasteboard + cmd+V to handle multi-line + UTF-8)
#     press cmd+enter (submit)
# - Capture screenshot before+after as evidence (saved to .aragora/dispatch-receipts/screenshots/)
# - Return exit 0 on detected submission, exit 1 if the UI state didn't change
```

Required tooling:

- macOS only (osascript). Add a platform-detect at entry: linux/windows → exit with helpful "not yet supported" message + fallback to mailbox.
- Accessibility permissions must be granted to Terminal.app / iTerm / whatever shell calls the script. First-run prompts the operator + writes a one-time setup receipt.

Tests: ≥6 — osascript subprocess invocation mocked; platform-detect; missing-prompt-file; thread-id resolution; accessibility-permission missing; screenshot capture.

Bounded: ≤90 min · single PR · pure bash + osascript + small Python helper for thread-id resolution.

**Honest caveat:** Codex Desktop's UI can change between releases. The osascript path needs occasional adjustment. The HTTP-API-on-localhost path (if discoverable) would be far more robust. Phase 3a (investigate the API) is the first checkpoint; if found, Phase 3 ships the API path and the osascript path becomes an opt-in fallback.

## Phase 4 — `scripts/droid_cli_inject.sh` (Factory web-CLI + local CLI)

**Two sub-paths:**

### Phase 4a — local Droid CLI in terminal

If the operator runs `droid` in a local terminal (not factory.ai web), it's a normal CLI process and can be tmux-wrapped. Detect this case at lane-claim time: if the Droid is launched via `agent_bridge.py launch` or `tmux_session_launcher.sh`, the contact method is `tmux:<name>`. Phase 2's `wake_agent.sh` handles it natively — no new code needed.

### Phase 4b — Factory web-CLI

The factory.ai web CLI is a hosted SaaS. To reach into it:

| Path | Verdict |
|---|---|
| Factory public API | Investigate. Factory has `droid exec` for non-interactive batch; if their API exposes sending a message to a live session, use it. Likely operator-account-scoped + requires API key. |
| Browser automation (Playwright) | Cross-platform, robust to UI changes if their selectors are stable. Adds a `playwright` dep — opt-in via extras. Slowest dispatch (cold-start ~5s). |
| osascript on macOS targeting Chrome/Safari window | Same as Codex Desktop osascript path but for the browser. Brittle, OS-specific, but no new deps. |
| Operator-side polling sidecar | A small `droid_inbox_poller.py` the operator runs once. Polls steering mailbox, prints `priority=blocking` messages to stderr loud enough to interrupt. Doesn't actually INJECT but gives notification. Cheap fallback. |

**Recommended:** Phase 4b ships the polling sidecar (`droid_inbox_poller.py`) as the MVP. Operator runs `python3 scripts/droid_inbox_poller.py --session <id>` once and it tails their inbox forever. Phase 4c can add Playwright + Factory API later as opt-in upgrades. The polling sidecar is honest about its limitations ("notify, don't inject") and works today without any external dependency.

Tests: ≥5 — happy path; priority-blocking interruption; missing inbox dir; multi-session concurrent polling; `--ack` flag moves message to `_acked/`.

Bounded: ≤45 min · single PR · pure stdlib.

## Phase 5 — Bootstrap sweeper for existing sessions

**Reality check:** at the time this plan lands, the lane registry will have many lanes from sessions claimed under the old (no `contact_method`) schema. New lanes can populate the field at claim time, but old ones can't be retroactively contacted.

**Sweeper:** `scripts/sweep_lane_contact_methods.py` runs once after Phase 1 + 2 land. For each `status=active` lane without `contact_method`:

- Inspect the lane's `worktree` against `~/.aragora/tmux-sessions/` — if a matching tmux log exists, infer `contact_method=tmux:<name>` (with verification by reading the log's most-recent timestamp).
- Inspect `~/.codex/sessions/` for a rollout matching the lane's `codex_thread_id` (if populated) — if found, infer `contact_method=osascript:codex-desktop:<thread>`.
- Inspect `~/.factory/background-processes.json` — if a process matches branch + cwd, infer `contact_method=factory-api:<session>` (placeholder pending Phase 4b).
- All other cases → `contact_method=mailbox-only`.

The sweeper runs in dry-run by default; `--apply` writes the inferred values via `claim_active_agent_lane.py` (no atomic-write race since it goes through the existing claim path).

Tests: ≥6 — each inference path; missing source files; ambiguous match (two tmux logs matching one worktree); dry-run vs apply; existing contact_method not overwritten; idempotent on re-run.

Bounded: ≤45 min · single PR · pure stdlib.

## Sequencing + cross-PR dependency graph

```
Phase 1 (contact_method on LaneRecord)
    │
    ├─→ Phase 2 (wake_agent.sh unified CLI)
    │       │
    │       ├─→ Phase 3 (codex_desktop_inject.sh)
    │       ├─→ Phase 4a (local Droid CLI — no new code; Phase 2 covers it)
    │       └─→ Phase 4b (droid_inbox_poller.py sidecar)
    │
    └─→ Phase 5 (bootstrap sweeper — runs once after Phase 1 lands)

Phase 6+ (opt-in upgrades): Factory API client, Codex Desktop HTTP API client,
Playwright fallback. Each ships as its own opt-in PR after the core works.
```

Estimated total: ~5 hours of bounded work across 5 PRs.

## What this WON'T solve

Even with all five phases shipped:

1. **Cross-machine fan-out.** The mailbox lives in `.aragora/operator-steering/` inside the local repo. Cross-machine still requires either git-push-of-the-mailbox or a separate sync surface (out of scope; possibly a Phase 7).
2. **Active resistance.** A Codex Desktop tab that's NOT focused, or a Droid web session that's NOT logged in on the operator's machine, can't be reached. Operator presence is assumed.
3. **Self-defending agents.** Nothing here stops a malicious agent from ignoring incoming messages. The model is cooperative.
4. **Live wake-up vs response.** `wake_agent.sh` injects a prompt. The agent's *response* is captured in the agent's own log / tmux pane — there's no synchronous-RPC reply channel. Two-way live RPC is a separate Phase 8+.

## Holds + safety

- Phase 1-2 are pure-stdlib + safe; ship as standard fan-out lanes.
- Phase 3 + Phase 4b touch OS-level automation (osascript / accessibility). First-run requires operator permission grant; document it loudly.
- Phase 4b's `droid_inbox_poller.py` runs as a persistent daemon. Operator opt-in via explicit `launchctl load` (no auto-install). Default exit on SIGINT.
- No held PR touches. No protected-file edits. Zero AI-key consumption from any phase.
- Each phase ships as its own draft PR with the standard lane-registry claim + receipt trio + journal append discipline.

## Open questions for operator

1. **Codex Desktop HTTP API.** Is there a localhost port open when Codex.app is running? If yes, what's the surface? This is the cleanest Phase 3 path. (Investigation step before committing to osascript.)
2. **Factory API scope.** Does the operator's Factory.ai account have API access for live-session sending? If yes, Phase 4b can ship as `factory_api_send.py` directly and skip the polling-sidecar fallback.
3. **Cross-machine.** Is cross-machine reach a real requirement? Or is "operator-on-one-machine, agents fan out on that machine" the actual usage pattern? The answer changes Phase 7.
4. **Daemon model.** For Phase 4b's polling sidecar, does the operator prefer (a) a long-lived background daemon, (b) a one-shot per session, or (c) a launchd-managed service?

Operator clarification on these accelerates execution. In the absence of answers, Phase 1-2 + 4b can ship without prejudice; Phase 3 must investigate the API first.

## Acceptance for full primitive

After all five phases ship and the sweeper runs once:

- Every entry in `.aragora/agent-bridge/lanes.json` has a non-null `contact_method`.
- `wake_agent.sh --lane <any-active-lane> --prompt "hello"` produces a delivery receipt and exits 0.
- The operator can ask "what's everyone doing right now?" via `agent_bridge.py operator-snapshot --json` and reach any one of them via `wake_agent.sh`.
- The four "weak/does-not-exist" capabilities are closed: Droid CLI (Phase 4a or 4b), Codex Desktop (Phase 3), universal dispatch (Phase 1+2+5), one-button wake-up (Phase 2).

## Receipt

This plan is itself the deliverable for P54-agent-dispatch-reach-plan. The five execution PRs are tracked as P55..P59 (or whatever uncontested numbers the namespace permits at claim time). Phase D of the agent-steering primitive (P52) and Phase E (P53) are unrelated parallel work and can ship before, during, or after this plan's phases.
