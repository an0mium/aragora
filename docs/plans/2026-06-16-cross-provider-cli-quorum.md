# Cross-Provider CLI Quorum (Tier-4 — preapproval required)

**Status:** **operator-preapproved 2026-06-16** (implementation authorized); the
PR still requires a **head-bound Western-only operator settlement to merge** ·
**Surface:** `aragora/swarm/quorum_evidence.py` — **merge-authority (Tier-4)**

> Reviewers run **sandboxed/read-only**: grok-build with `--sandbox read-only`,
> `agy` with `--sandbox` — they cannot write/exec in the merge-gate cwd.

## Why

The merge-quorum gate needs 2 distinct model families. Today only two families
run via **subscription CLI** (`claude` → `claude -p`, `openai` → `codex exec`);
`grok`/`gemini` only run via the **API path** (`_run_api_agent`, needs API keys /
MFA). So when one CLI provider hits its cap, the gate stalls.

This is not hypothetical — on 2026-06-16 the **codex sub hit its 5-hour usage
limit mid-merge** ("try again at 3:00 PM"), leaving `claude + codex` unable to
form a quorum even though `grok-build` and `antigravity` were authed and idle.

## Change

Add two subscription-CLI reviewer runners and route the corresponding families
to them when the CLI is present, mirroring how `_run_openai_reviewer` already
prefers the Codex CLI over the API:

- `grok` family → `~/.grok/bin/grok --no-plan -p <prompt>` (Grok Build; resolve
  the explicit path to avoid the broken legacy `grok` on PATH — same
  `_resolve_grok_build_bin` logic as `GrokBuildAgent`).
- `gemini` family → `agy -p <prompt>` (Antigravity CLI, Google AI Ultra).
- Both: fall back to the existing `_run_api_agent` path if the CLI is absent, so
  behavior is unchanged where the CLIs aren't installed.

Result: the gate can form a 2-family quorum from **any two** of
`claude · codex · grok-build · antigravity` — so one provider capping (codex
today) no longer blocks merges.

## Safety invariants (unchanged — this only adds reviewer *backends*)

- **No change to counting rules.** `FAMILY_PROVIDERS` is untouched; families stay
  distinct; **Fusion stays excluded** (blend → would double-count). The 2-family
  minimum, tier gating, and `SETTLEMENT_TIER_FLOOR` are unchanged.
- **No change to settlement/auth.** `settle_one_pr` remains the sole merge
  authority; Tier-3/4 still require operator settlement.
- **Reviewers are read-only** (`--sandbox read-only` / MCP-disabled where
  supported), grounded on the exact head, same evidence-lint before posting.
- **Default-preserving:** absent a CLI, the family uses the current API path.

## Why this is Tier-4 (needs preapproval before implementation AND merge)

`quorum_evidence.py` is a **merge-authority surface** under
`docs/REVIEW_AUTHORITY_PRINCIPLES.md`: code that decides what evidence counts
toward a merge. Even though this change only adds reviewer backends (not counting
logic), editing this file is a merge-authority self-modification → requires
human preapproval before implementation and a head-bound operator settlement on
the PR (Western-only counted quorum).

## Plan once preapproved

1. Add `_run_grok_build_cli` + `_run_antigravity_cli` (mirror `_run_codex_openai_cli`).
2. Route `grok`/`gemini` in the family dispatch to CLI-first, API-fallback.
3. Tests: each runner builds the right argv; family dispatch prefers CLI when
   present; counting rules + Fusion-exclusion unchanged (regression).
4. Land as its own Tier-4 PR with operator settlement.

## Immediate payoff

`claude + grok-build` (or `claude + antigravity`) can merge **#8481 now**, without
waiting for codex's ~3pm reset — and the gate gains permanent cross-provider
resilience.
