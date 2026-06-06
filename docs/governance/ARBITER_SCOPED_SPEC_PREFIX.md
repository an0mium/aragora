# Scoped agent-owned spec prefix for the autonomous merge-arbiter

Status: **Tier-4 merge-authority change — requires exact-head human settlement.**
Decision owner: repo founder. This PR prepares the change; it must not be merged
without explicit, repo-visible operator settlement on the exact head SHA.

## Decision

Add **`aragora/spec/`** to the merge-arbiter's automation-owned branch prefixes
(`aragora/swarm/merge_arbiter.py::AUTOMATION_BRANCH_PREFIXES`) so that
**agent-produced spec work** (e.g. a ground-truth-integrity benchmark spec) can be
judged and settled by the autonomous frontier-LLM quorum harness — the same path
that already serves `codex/`, `factory/`, and `aragora/boss-harvest/`.

Bare **`spec/` is deliberately excluded.** `spec/` is a generic, human-natural
namespace for design/WIP; making it arbiter-eligible would let any green+quorum'd
human design draft become an auto-merge candidate. Scoping to the `aragora/`
automation namespace (which already hosts `aragora/boss-harvest/`) keeps the
ownership boundary intact: agents publish autonomy-bound specs under
`aragora/spec/<topic>-<date>`; humans keep design drafts on bare `spec/`.

## Why scoped, not bare (the rejected option)

The branch-prefix gate is a **merge-authority boundary**, not a convenience. The
autonomous arbiter only *considers* PRs whose head branch is automation-owned.
Widening it to bare `spec/` is a broad, repo-wide expansion into human design
space; the scoped `aragora/spec/` form delivers the same autonomy for agent work
with a narrow, named blast radius.

## What this changes — and what it does NOT

- **Candidacy only.** The prefix gate decides whether the arbiter *looks* at a PR.
  It does **not** bypass any quality gate. An `aragora/spec/` PR still requires:
  all required checks green, a real ≥2-family model quorum, tier classification,
  the Tier 0-2 authorization packet / recorded merge-on-green, and dogfood evidence
  for proof surfaces — exactly as for any other automation branch.
- **Drafts stay manual.** Draft PRs are not auto-promoted/merged. Keeping an
  `aragora/spec/` PR in draft is the manual hold lever.
- **Draft auto-promotion is untouched.** `boss_loop.py::_draft_promotion_ownership`
  (which can *un-draft* PRs) intentionally matches only `aragora/boss-harvest/issue-`
  and `codex/swarm-`; it is **not** modified here, so spec drafts are never
  auto-undrafted.
- **Launch-arg safety.** `_normalize_branch_prefixes` maps any bare `spec` / `spec/`
  / `aragora/spec` launch argument to the scoped `aragora/spec/`, so an operator
  cannot accidentally widen the gate to human `spec/` work via `--branch-prefix`.
- **Ownership class.** `aragora/spec/` classifies as `queue-owned` (like `codex/` /
  `factory/`), not `boss-owned` (reserved for `aragora/boss-harvest/`).

## Scope of this change

`aragora/swarm/merge_arbiter.py` only (the daemon the request named). The separate
proof-first harness (`scripts/run_proof_first_shift.py`) keeps its own prefix list;
extending it to `aragora/spec/` is a deliberately separate follow-up decision, kept
out of this PR to bound the Tier-4 blast radius.

## Usage

Name autonomy-bound agent spec branches `aragora/spec/<topic>-<YYYYMMDD>`. They then
flow through the existing autonomous Tier-1/2 quorum-judged settlement path, subject
to all gates above. Human design drafts remain on bare `spec/` and stay manual.

## Tier-4 settlement

This is merge-authority self-modification (per `docs/AGENT_OPERATING_CONTRACT.md` and
the elves-aragora governance gate): prepared autonomously, **merged only** after
explicit exact-head operator settlement
(`python3 scripts/settle_tier4_pr.py --check --pr <N> --head <SHA>` + repo-visible
authorization).
