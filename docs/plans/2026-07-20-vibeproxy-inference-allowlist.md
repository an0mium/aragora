# VibeProxy inference-site inventory and static allowlist

## Mission

Define the legal inference-routing surface before any additional VibeProxy runtime routing is added. This bounded PR will inventory the current production OpenAI and Anthropic inference call sites, classify each as `proxy-eligible` or `direct-only`, and add a deterministic static check that fails when a new site or forbidden endpoint is introduced without an explicit classification.

This is the prerequisite work unit recommended by the July 20 Claude Fable 5 consult for issue #9409. The next units remain separate PRs: exact-match OpenAI Chat/Responses routing, then endpoint authentication/pinning before broader provider expansion.

## Scope

### In scope

- A reviewable allowlist artifact containing stable path/symbol anchors, provider, policy classification, and rationale for every `direct-only` entry.
- A deterministic, standard-library static scanner for production OpenAI/Anthropic client construction and hardcoded inference endpoints.
- Regression tests proving current-main inventory completeness, rejection of unclassified additions, and rejection of port 8317.
- Documentation for maintaining and validating the inventory.

### Out of scope

- Runtime request routing or transport-policy behavior changes.
- Sending inference requests, burn-in calls, reviewer shadow calls, or countable evidence.
- VibeProxy server changes, endpoint authentication/pinning implementation, metrics, or trace metadata.
- CI workflow, governance, merge-quorum, settlement, branch-protection, public API, or SDK changes.
- Any merge. The user reviews and merges the PR.

## Batch 1: Inventory and enforcement

### Tasks

- [x] Survey existing inference construction patterns and existing audit/checker conventions.
- [x] Add the static allowlist and deterministic scanner without adding dependencies.
- [x] Classify every discovered current production site and require a rationale for direct-only entries.
- [x] Add focused scanner/manifest tests and direct-path policy assertions.
- [x] Document the maintenance command and classification rules in checker help and the generated manifest metadata.

### Acceptance criteria

- [x] Current-main inventory is an exact deterministic match with no unclassified or stale entries.
- [x] A synthetic new OpenAI/Anthropic inference site fails the checker until classified.
- [x] Port 8317 is rejected in scanned source and allowlist data.
- [x] CI, production, credential validation, public gateway, and evidence-related paths remain explicitly direct-only where applicable.
- [x] `scripts/consult_claude.py` is the only existing non-test `ModelTransportPolicy` consumer and is classified deliberately.
- [x] Focused tests, changed-file type/lint gates, charter compliance, and automation preflight pass.
- [ ] Fresh independent review finds no P0-P2 blocker and the final cumulative diff is review-ready.

### Docs likely touched

- A focused VibeProxy/inference-routing maintenance document or checker help text.
- This plan.

### Risk

Static discovery can become noisy or line-number-fragile. The implementation must use stable path/symbol anchors, constrain discovery to intentional production roots, and test both missing and stale inventory entries.

## Non-negotiables

- VibeProxy remains a transport, never a reviewer family.
- Never select, permit, or normalize port 8317.
- No silent semantic model substitution and no fallback after output begins.
- Direct-only surfaces stay direct: CI, production, credential validation, public gateways, and merge/evidence authority.
- No inference requests, no later #9409 runtime units, no governance changes, and no merge in this PR.

## Test strategy

- Focused static-checker tests under `tests/scripts/` or the existing audit test surface.
- Direct invocation of the checker against the branch tip in both human and JSON modes if the repository pattern supports both.
- Changed-file pre-commit/mypy checks, `scripts/check_charter_compliance.py`, and `scripts/automation_pr_preflight.sh origin/main HEAD`.
- Broad regression proof before final readiness, proportionate to this additive tooling-only change.

## Batch sizing

```yaml
team-size: 4
sprint-length: 2 weeks
```

## Dependency order after this PR

1. Exact-match OpenAI Chat/Responses routing through the central policy, keeping web-search and untested capabilities direct.
2. Endpoint authentication or pinning before broad automatic rollout on shared hosts.
3. Native Anthropic and proxy-backed Grok/Gemini/Kimi adapters, then low-cardinality observability.
4. Seven-day/100-call burn-in, 20 non-countable reviewer shadows, and only then any separately authorized Tier-4 evidence change.
