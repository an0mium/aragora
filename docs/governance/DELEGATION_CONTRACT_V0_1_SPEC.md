# Aragora Delegation Contract v0.1 — Specification

**Status:** draft v0.1 — schema-first, no autonomous behavior change yet.
**Author:** claude-B061F80D (synthesis of operator + codex + Factory review)
**Date:** 2026-05-19

## Why this exists

Today Aragora can spawn autonomous agents (tmux-managed Claude workers,
Codex CLI workers, Droid CLI workers, Factory droid workers). Each spawn
extends a chain of authority that ultimately roots in a human operator,
but the chain itself is implicit: there's no machine-checkable artifact
that says "this agent was authorized to do this action, in this scope,
in service of this goal, for this much budget, by whom."

Without that artifact, three failure modes are systemic rather than
incidental:

1. **Authority laundering.** Child agent inherits permissions parent
   never had explicit operator approval to delegate.
2. **Scope drift.** Operator authorized "ship the reach plan"; agent
   spawns a chain of work that quietly expands to "refactor all of
   `scripts/`."
3. **Unbounded cascade.** Parent agent spawns child agents that spawn
   their own children — no formal limit, no consolidated budget, no
   way to revoke mid-cascade.

The Anthropic-side classifier polarity addresses these reactively by
denying actions that look risky. The Factory-side auto-mode polarity
addresses these by trusting the agent and counting dollars after the
fact. **Neither is sufficient: scope (what) and budget (how much) are
two independent dimensions, and a real protocol needs both.**

The Delegation Contract makes scope + budget + goal + chain auditable
as a single signed-or-shouldhavebeensigned artifact that flows with
each lane claim and each subagent dispatch.

## Design influences

This v0.1 synthesizes:

- **Object-capability security** (E lang, KeyKOS, Genode): authority
  as unforgeable tokens passed explicitly; child scope must be subset
  of parent scope.
- **Ethereum gas**: per-op cost, bounded budget, atomic revert.
- **AWS IAM AssumeRole with session policies**: production-proven
  cert narrowing across permission boundaries.
- **SLSA / in-toto**: cryptographic provenance attestations along the
  build chain (deferred to a later contract version; v0.1 is
  schema-only).
- **OAuth 2.0 scopes**: scope-set + expiry + audience model.
- **Aragora's existing `RiskBudget` / `ToolCapability` / `Policy`
  primitives**: this contract composes them rather than replacing
  them.

Operator review notes that shaped v0.1:

- **codex**: "extend existing surfaces, not invent a parallel
  certificate/orchestration stack"; "schema-first and enforceable at
  lane-claim/dispatch boundaries"; "rename 'Capability Certificate'
  to 'Delegation Contract' for v0.1 unless cryptographic signing is
  actually implemented in the same PR."
- **Factory**: "the orthogonality of ocap (scope) and gas (budget) is
  the protocol's most important contribution"; "predicate oracle
  decoupling — the largest unaddressed gap"; "trust-adjusted budgets
  via ELO"; "continuous adversarial verification instead of periodic
  spot-checks"; "cross-family cert bridging"; "three-tier
  reversibility: pause / halt / revoke".

## v0.1 Schema

### `DelegationContract`

```python
@dataclass(frozen=True)
class DelegationContract:
    # --- Identity ---
    contract_id: str              # ULID-like, monotonic
    schema_version: str           # "aragora-delegation-contract/0.1"

    # --- Authority chain ---
    root_intent_id: str           # The original human-rooted intent
    parent_contract_id: str | None  # None iff root; else MUST resolve to
                                    # a contract whose scope contains
                                    # this one's
    delegator: str                # session id of issuer
    delegatee: str                # session id of subject
    max_depth: int                # subagents this can issue must have
                                  # max_depth = this.max_depth - 1; 0
                                  # means leaf, cannot spawn

    # --- Goal binding ---
    goal_id: str                  # Resolves to a GoalSpec

    # --- Scope (ocap dimension) ---
    allowed_actions: frozenset[str]
        # e.g. {"read:*", "write:branch:claude/*",
        #       "write:draft-pr:*", "spawn:subagent"}
    denied_actions: frozenset[str]
        # explicit deny list; takes precedence over allowed
    allowed_surfaces: AllowedSurfaces
        # PRs, branches, worktrees, file globs
    destructive_action_policy: Literal["deny", "human-only", "allow"]
        # default "human-only" — destructive actions require explicit
        # human approval regardless of any other field

    # --- Budget (gas dimension) ---
    budget: ContractBudget        # see below

    # --- Lifecycle ---
    issued_at: str                # ISO-8601 UTC
    expires_at: str               # ISO-8601 UTC, must be in future at
                                  # issue time
    revocation_check_uri: str | None
        # Where to check liveness; v0.1 uses local file path

    # --- Progress gating ---
    progress_predicates: list[str]
        # References into the predicate oracle namespace (R01-style)
    stale_threshold_minutes: int  # No-progress-for-N halt threshold

    # --- v0.1 stub for v0.2 signing ---
    signature: str | None         # Always None in v0.1; reserved for
                                  # HMAC/ed25519 in v0.2
```

### `AllowedSurfaces`

```python
@dataclass(frozen=True)
class AllowedSurfaces:
    pr_numbers: frozenset[int]          # specific PRs (empty = any in branches)
    branch_globs: frozenset[str]        # e.g. {"claude/*", "claude/R*"}
    worktree_globs: frozenset[str]      # e.g. {".worktrees/codex-auto/claude-*"}
    file_globs: frozenset[str]          # e.g. {"scripts/wake_agent.sh", "tests/scripts/test_wake_agent.py"}
    deny_file_globs: frozenset[str]     # e.g. {"CLAUDE.md", "aragora/__init__.py"}
```

### `ContractBudget`

Composes Aragora's existing `RiskBudget` with additional fan-out and
artifact-output caps:

```python
@dataclass(frozen=True)
class ContractBudget:
    risk_budget: RiskBudget           # existing aragora.policy.risk primitive
    max_wall_clock_minutes: int       # hard real-time cap
    max_subagents_spawned: int        # cascade fan-out cap
    max_prs_opened: int               # output cap
    max_commits_to_main: int          # main-branch write cap
    max_api_dollars: float            # spend cap
    max_lane_claims: int              # registry-write cap
```

### `GoalSpec`

```python
@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    schema_version: str               # "aragora-goal-spec/0.1"
    owner: str                        # human GitHub login or operator id
    approved_at: str                  # ISO-8601 UTC
    description: str                  # human-readable summary
    acceptance_criteria: list[AcceptanceCriterion]
    progress_metric: Literal[
        "fraction_of_AC_satisfied",
        "all_AC_satisfied",
        "weighted_AC",
    ]
    completion_predicate: str         # oracle predicate; must be true to
                                      # consider goal done
    anti_signals: list[str]           # named anti-signal evaluators
    max_delegation_depth: int         # ceiling for any contract issued
                                      # under this goal
```

### `AcceptanceCriterion`

```python
@dataclass(frozen=True)
class AcceptanceCriterion:
    ac_id: str                    # e.g. "AC1"
    predicate: str                # oracle string, e.g. "pr_merged(7336)"
    weight: float = 1.0           # for weighted_AC metric
    description: str = ""         # human-readable
```

## Monotonic narrowing rules

When `parent_contract` issues `child_contract`, ALL of the following
must hold; violations are rejected at issue time:

| Rule | Constraint |
|---|---|
| Goal | `child.goal_id == parent.goal_id` (same intent) |
| Depth | `child.max_depth == parent.max_depth - 1` |
| Actions | `child.allowed_actions ⊆ parent.allowed_actions` |
| Actions | `child.denied_actions ⊇ parent.denied_actions` (child denies at least everything parent denies) |
| Surface PRs | `child.allowed_surfaces.pr_numbers ⊆ parent.allowed_surfaces.pr_numbers` (or both empty) |
| Surface branches | every glob in `child.allowed_surfaces.branch_globs` is contained by some glob in `parent.allowed_surfaces.branch_globs` |
| Surface worktrees | analogous |
| Surface files | analogous; `child.deny_file_globs ⊇ parent.deny_file_globs` |
| Destructive | `child.destructive_action_policy` is "deny" or matches parent (cannot widen) |
| Budget | every field of `child.budget` ≤ corresponding field of `parent.budget.remaining` (budget is debited from parent, not duplicated) |
| Wall clock | `child.expires_at ≤ parent.expires_at` |
| Stale threshold | `child.stale_threshold_minutes ≤ parent.stale_threshold_minutes` (children fail faster than parents) |

These are enforced by `DelegationContract.narrow_for_child(...)` at
issue time. v0.1 ships the validator + tests; the actual lane-registry
hookup is Stage 2.

## Predicate oracle

The single biggest insight from Factory's review: **predicates must be
evaluated by deterministic non-LLM code, not by the agent itself or by
spot-check models.**

Without this, ACAP becomes "debugging-by-LLM all the way down."

v0.1 ships a small predicate oracle with explicit named evaluators:

| Predicate | Implementation |
|---|---|
| `pr_merged(N)` | `gh pr view N --json mergedAt --jq '.mergedAt'` → non-null |
| `pr_open(N)` | `gh pr view N --json state` → state == "OPEN" |
| `tests_pass(path)` | `pytest path -q --tb=no` exits 0 |
| `file_exists(path)` | `os.path.exists(path)` |
| `branch_exists(name)` | `git branch --list name` non-empty OR `git ls-remote --heads origin name` non-empty |
| `commit_landed(sha)` | `git merge-base --is-ancestor sha origin/main` returns 0 |
| `issue_closed(N)` | `gh issue view N --json state` → state == "CLOSED" |

Each predicate has a deterministic Python implementation in
`aragora/policy/predicate_oracle.py`. The oracle returns:

```python
@dataclass(frozen=True)
class PredicateResult:
    predicate: str               # the original string
    satisfied: bool              # boolean outcome
    evidence: str                # what was checked (e.g. "merged_at=2026-05-19T18:13:28Z")
    evaluated_at: str            # ISO-8601 UTC
    evaluator: str               # which evaluator ran it
    error: str | None            # if the oracle failed (e.g. gh unavailable)
```

**Non-oracle predicates are not acceptance criteria.** If a desired
outcome cannot be reduced to a deterministic check, it goes in the
human-review notes section of the goal spec, not in
`acceptance_criteria`.

## Destructive-action policy (always human-only in v0.1)

Regardless of any other field, the following action classes ALWAYS
require explicit human approval in v0.1:

- `force-push:*`
- `delete:branch:*`
- `delete:worktree:*` (the canonical helper paths are allowed; raw `rm -rf` is not)
- `kill:process:*`
- `merge:*`
- `label:*`
- `edit:protected:*` (see CLAUDE.md "Protected Files" list)
- `unshallow:*`
- `gpg-bypass:*`

These are encoded in `DEFAULT_DESTRUCTIVE_ACTIONS` and a contract
cannot weaken them (the validator rejects any contract whose
`allowed_actions` includes a destructive action without
`destructive_action_policy == "human-only"` + an explicit
`human_approval_token` field, which v0.1 does NOT ship — so v0.1
effectively forbids destructive actions across the board).

## What v0.1 does NOT include

Explicit non-goals for this PR:

- HMAC / ed25519 signing of contracts (deferred to v0.2)
- Lane-registry integration (`delegation_contract_id` column on
  LaneRecord) — that's Stage 2, separate PR
- Continuous adversarial spot-check execution — Stage 4
- ELO trust-multiplier on budget — Stage 4 (requires receipt feedback
  loop)
- Cross-family adapter (`--parent-cert` flag on `launch_lane.sh`) —
  Stage 4
- Three-tier reversibility orchestration (pause / halt / revoke) —
  Stage 3
- Push-revocation kill-switch file watcher — Stage 3

v0.1 ships ONLY: the spec doc, the joint dataclass module, the
predicate oracle, and tests. No autonomous worker behavior change.

## Path to v0.2+

| Version | Adds |
|---|---|
| v0.1 (this PR) | Schema + validator + predicate oracle + tests; no behavior change |
| v0.2 | Lane registry `delegation_contract_id` field; `claim_active_agent_lane.py --parent-contract` enforcement |
| v0.3 | Progress-ledger periodic predicate evaluation; stall-detection thresholds |
| v0.4 | HMAC signing via existing `aragora.security.context_signing`; receipt schema references contract_id |
| v0.5 | Continuous adversarial check (Haiku-tier model alongside worker); debate panel escalation via existing Arena |
| v0.6 | ELO trust multiplier on effective budget |
| v0.7 | Three-tier reversibility (pause / halt / revoke) with push-revocation kill switch |
| v0.8 | Cross-family adapter for Factory / Codex / Droid harnesses |
| v1.0 | All of the above hardened, audited, default-on for new lanes |

Each stage is a separately-reviewable PR. v0.1 must be visible and
tested before any of v0.2+ is meaningful.

## Open questions for v0.2+ review

1. **Where does the goal spec live?** `docs/goals/<goal-id>.yaml` (tracked in repo) or `.aragora/goals/<goal-id>.json` (operator-local)? v0.2 needs to decide; v0.1 is schema-only so it doesn't matter yet.
2. **Predicate caching.** Some predicates (e.g. `pr_merged`) are expensive. v0.3 progress ledger needs a cache policy.
3. **Identity verification.** v0.1 takes `delegator` / `delegatee` as plain strings; v0.4 signing makes them verifiable.
4. **Cross-family delegator naming.** Factory worker spawned by Claude session — what's the delegator? The Claude session or the Factory harness? v0.8 needs to settle this.

## References

- `aragora/policy/risk.py` — `RiskBudget`, `RiskLevel`, `BlastRadius`, `RiskActionRecord`
- `aragora/policy/tools.py` — `ToolCapability`, `ToolCategory`
- `aragora/policy/engine.py` — `Policy`, `PolicyEngine`, `PolicyDecision`, `PolicyResult`
- `aragora/security/context_signing.py` — HMAC signing primitives (deferred to v0.4)
- `aragora/ranking/elo.py` — agent skill ELO (deferred to v0.6 trust multiplier)
- `scripts/claim_active_agent_lane.py` — lane registry mutation surface (v0.2 integration point)
- `docs/AGENT_OPERATING_CONTRACT.md` — protected, but defines current operator authority model
- `PR #7327` — agent-dispatch reach plan; depends on this contract for v0.8 cross-family bridging
