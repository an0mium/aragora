# VibeProxy allowlist run learnings

## Repo conventions

- 2026-07-20: New #9409 units must remain separate bounded PRs; inventory/allowlist precedes runtime routing so the legal surface is explicit before policy consumers expand.

## Validation and tooling

- 2026-07-20: Claim both the agent-bridge lane and `check_work_lease.py` branch lease before publishing; re-read operator steering before mutation.

## Review heuristics

- 2026-07-20: Static inventory should use stable path/symbol anchors, not line numbers, and must detect both new unclassified sites and stale manifest entries.

## Product and domain invariants

- 2026-07-20: VibeProxy is a transport, not a reviewer family; port 8317 is forbidden; CI, production, credential validation, public gateways, and evidence/merge authority remain direct-only.

## Known traps

- 2026-07-20: A Fable consult is strategy input only. Re-ground main, mailbox, owner, issue overlap, and branch lease before executing its recommendation.
- 2026-07-20: Fail-closed scanners must surface parse/read failures, track constructor aliases, and avoid suffix-only method matches; a generated manifest still needs stable review lines.
- 2026-07-20: Port prohibition must cover normalized textual equivalents such as `:08317`, while exempting only the exact central declaration of the forbidden value.
- 2026-07-20: SDK call discovery should follow simple import/type/assignment/factory provenance; receiver-name substrings create both false positives and false negatives.
- 2026-07-20: Generated policy templates must advertise when their safe default differs from reviewed committed exceptions so regeneration cannot silently erase policy.
