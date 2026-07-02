---
name: consult-fable
description: Bounded advisory consult of Claude Fable 5 from inside the aragora repo. Use when you are stuck on prioritization ("what should I work on next?"), need a second opinion on a design or a reviewer deadlock, want a plan sanity-checked, or were asked to "ask Claude / Fable" something. Runs scripts/consult_claude.py with a per-attempt timeout (default 600s), strict empty MCP config for CLI calls, and fail-closed backend attempts. Advice is input, not authority — it cannot approve merges, settle quorum, or override gates.
license: MIT
compatibility: Works with Codex (.agents/skills), Claude Code (.claude/skills), and any Agent Skills platform. Requires python3; uses the local `claude` CLI when present, else the Anthropic API via aragora's secrets manager.
metadata:
  author: Synaptent (aragora)
  version: "1.0.0"
  argument-hint: The question to ask, or a path to a prompt file.
---

# Consult Fable (bounded Claude Fable 5 advisory)

Ask Claude Fable 5 a question and get an answer back through bounded backend
attempts.
This replaces ad-hoc `timeout 120 claude -p "..."` calls, which hang or expire
with no output. The tool passes the prompt via stdin, forwards `--model`,
disables local MCP servers for CLI attempts, enforces a per-attempt timeout,
and falls back (claude CLI on
`claude-opus-4-8`, then the Anthropic Messages API) if the primary attempt
fails. The timeout is per backend attempt, not a single process-wide wall-clock
cap; use an external wrapper if your conductor cycle needs a stricter total
budget.

## How to invoke

From the repo root (any worktree):

```bash
# Inline question, defaults: model claude-fable-5, timeout 600s
python3 scripts/consult_claude.py "One-paragraph question with the live state inlined."

# Long prompt from a file, machine-readable result
python3 scripts/consult_claude.py --prompt-file /tmp/question.md --json

# Bigger budget for a hard question
python3 scripts/consult_claude.py --timeout 900 --prompt-file /tmp/question.md
```

`--json` returns `{ok, model, backend, elapsed_s, text, attempts}`. Exit codes:
`0` ok, `2` timed out, `3` no prompt, `4` all backends failed.

If `scripts/consult_claude.py` does not exist in your checkout yet, use an
installed user skill copy such as `~/.codex/skills/consult-fable/consult_claude.py`.
The tracked repo skill itself expects the tracked repo script.

## Writing a good consult prompt

Fable has **no access to your session state**. Inline everything it needs:

- The decision you are facing and the 2-4 options you see.
- Live facts, verified this cycle: PR numbers with head SHAs, gate/quorum
  state, dissent findings, owner state. Never paste stale transcript claims.
- Constraints that bind you (operating contract tier limits, no `--admin`,
  shared-root read-only, anti-treadmill rules).
- Ask for a **single recommendation with reasoning**, not a survey.

For "what should I work on next?" consults, a good shape is: list the
candidate progress units with one line of live state each, state the
anti-treadmill rules that apply, and ask Fable to pick exactly one and say why.

## Hard rules

1. **Advisory only.** The answer is one input to your decision. It is never
   authority to merge, settle, post evidence, or bypass a gate. Verify any
   factual claim in the answer against live repo state before acting on it.
2. **One consult per decision.** Do not loop consults on the same question;
   if the answer doesn't unblock you, the blocker is information, not advice.
3. **Bounded always.** Keep the default 600s per-attempt timeout or raise it
   explicitly; never wrap this tool in an outer retry loop.
4. **No secrets in prompts.** Inline repo state, not credentials or tokens.

## Failure handling

- Exit 2 (timeout) or 4 (all backends failed): report the consult as
  unavailable and proceed with your own judgment. Do not retry more than once.
- If only the API backend fails with a missing-key error, the local `claude`
  CLI is the expected path; check `which claude` before concluding anything.
