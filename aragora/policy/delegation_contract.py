"""Aragora Delegation Contract v0.1 — schema + validator.

This module ships the smallest durable artifact of the Delegation Contract
work: the dataclasses (DelegationContract, GoalSpec, AcceptanceCriterion,
ContractBudget, AllowedSurfaces) plus a monotonic-narrowing validator at
parent → child issue time.

No autonomous worker behavior changes in v0.1. The lane registry hookup
is Stage 2 (separate PR). The signing layer is v0.4. See
``docs/governance/DELEGATION_CONTRACT_V0_1_SPEC.md`` for the full
roadmap.

Design influences (per spec doc):
- Object-capability security: child scope ⊆ parent scope (enforced here)
- Ethereum gas: ContractBudget composes existing RiskBudget + fan-out caps
- AWS IAM AssumeRole with session policies: time-bounded delegation
- Factory's review: deterministic predicate oracle separate from this
  module so contract logic is itself testable without LLM dependencies

Operator review notes:
- codex: rename "Capability Certificate" → "Delegation Contract" for
  v0.1 unless cryptographic signing ships in the same PR; v0.1 does
  NOT ship signing, so the rename is honored.
- Factory: predicate evaluation is a separate non-LLM oracle module
  (``aragora.policy.predicate_oracle``); this module does not reach
  into the oracle directly — the goal spec references predicate
  strings and the caller evaluates them.

Pure stdlib. No new pip deps.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Iterable, Literal

from .risk import RiskBudget

CONTRACT_SCHEMA_VERSION = "aragora-delegation-contract/0.1"
GOAL_SPEC_SCHEMA_VERSION = "aragora-goal-spec/0.1"

DestructivePolicy = Literal["deny", "human-only", "allow"]
ProgressMetric = Literal["fraction_of_AC_satisfied", "all_AC_satisfied", "weighted_AC"]

# Actions that ALWAYS require explicit human approval, regardless of contract
# scope. v0.1 effectively forbids these because the contract format has no
# "human_approval_token" carrier yet.
DEFAULT_DESTRUCTIVE_ACTIONS: frozenset[str] = frozenset(
    {
        "force-push:*",
        "delete:branch:*",
        "delete:worktree:*",
        "kill:process:*",
        "merge:*",
        "label:*",
        "edit:protected:*",
        "unshallow:*",
        "gpg-bypass:*",
    }
)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class ContractValidationError(ValueError):
    """Raised when a contract or a parent→child narrowing fails validation."""


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriterion:
    """A single verifiable predicate for goal completion."""

    ac_id: str
    predicate: str  # predicate-oracle string, e.g. "pr_merged(7336)"
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class GoalSpec:
    """The "why" — what we're trying to accomplish, expressed as
    deterministically-verifiable predicates.

    Separate from DelegationContract because:
    - one goal may span multiple contracts (e.g. parent + multiple children
      all working toward the same goal)
    - goals are signed by humans; contracts can be issued by agents within
      a delegation chain
    """

    goal_id: str
    schema_version: str
    owner: str  # human GitHub login or operator id
    approved_at: str  # ISO-8601 UTC
    description: str
    acceptance_criteria: list[AcceptanceCriterion]
    progress_metric: ProgressMetric = "fraction_of_AC_satisfied"
    completion_predicate: str = ""  # oracle predicate; if empty, defaults
    # to "all AC satisfied"
    anti_signals: list[str] = field(default_factory=list)
    max_delegation_depth: int = 3

    def validate(self) -> None:
        """Sanity-check the goal spec. Raises ContractValidationError."""
        if not self.goal_id:
            raise ContractValidationError("goal_id must not be empty")
        if self.schema_version != GOAL_SPEC_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unexpected schema_version: {self.schema_version!r}; expected "
                f"{GOAL_SPEC_SCHEMA_VERSION!r}"
            )
        if not self.acceptance_criteria:
            raise ContractValidationError(
                "acceptance_criteria must not be empty — a goal with no verifiable "
                "acceptance criteria cannot drive progress evaluation"
            )
        ac_ids = [ac.ac_id for ac in self.acceptance_criteria]
        if len(set(ac_ids)) != len(ac_ids):
            raise ContractValidationError(f"duplicate ac_id in goal: {ac_ids}")
        if self.max_delegation_depth < 0:
            raise ContractValidationError(
                f"max_delegation_depth must be >= 0; got {self.max_delegation_depth}"
            )


@dataclass(frozen=True)
class AllowedSurfaces:
    """Scope dimensions: what surfaces this contract permits acting on.

    Empty sets mean "no restriction beyond the action filter" — but if
    the action requires a specific surface (e.g. write:branch:NAME) and
    the surface is not allowed, the action is denied.
    """

    pr_numbers: frozenset[int] = frozenset()
    branch_globs: frozenset[str] = frozenset()
    worktree_globs: frozenset[str] = frozenset()
    file_globs: frozenset[str] = frozenset()
    deny_file_globs: frozenset[str] = frozenset()

    def is_branch_allowed(self, branch: str) -> bool:
        if not self.branch_globs:
            return True
        return any(fnmatch.fnmatchcase(branch, g) for g in self.branch_globs)

    def is_file_allowed(self, path: str) -> bool:
        # deny wins
        if any(fnmatch.fnmatchcase(path, g) for g in self.deny_file_globs):
            return False
        if not self.file_globs:
            return True
        return any(fnmatch.fnmatchcase(path, g) for g in self.file_globs)


@dataclass(frozen=True)
class ContractBudget:
    """The "how much" — gas dimension. Composes existing RiskBudget."""

    risk_budget: RiskBudget
    max_wall_clock_minutes: int = 60
    max_subagents_spawned: int = 0
    max_prs_opened: int = 1
    max_commits_to_main: int = 0
    max_api_dollars: float = 1.00
    max_lane_claims: int = 1

    def validate(self) -> None:
        for field_name in (
            "max_wall_clock_minutes",
            "max_subagents_spawned",
            "max_prs_opened",
            "max_commits_to_main",
            "max_lane_claims",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ContractValidationError(
                    f"ContractBudget.{field_name} must be >= 0; got {value}"
                )
        if self.max_api_dollars < 0:
            raise ContractValidationError(
                f"max_api_dollars must be >= 0; got {self.max_api_dollars}"
            )
        if self.risk_budget.total < 0 or self.risk_budget.spent < 0:
            raise ContractValidationError("risk_budget total/spent must be >= 0")

    def child_fits(self, child: "ContractBudget") -> tuple[bool, str]:
        """Whether `child` budget fits within `self.remaining` capacity.

        Returns (fits, reason). Used by parent→child narrowing validation.
        """
        # Wall clock + counts must each be ≤ parent's
        for field_name in (
            "max_wall_clock_minutes",
            "max_subagents_spawned",
            "max_prs_opened",
            "max_commits_to_main",
            "max_lane_claims",
        ):
            parent_v = getattr(self, field_name)
            child_v = getattr(child, field_name)
            if child_v > parent_v:
                return (
                    False,
                    f"child.{field_name}={child_v} exceeds parent.{field_name}={parent_v}",
                )
        if child.max_api_dollars > self.max_api_dollars:
            return (
                False,
                f"child.max_api_dollars={child.max_api_dollars} exceeds parent={self.max_api_dollars}",
            )
        # Risk budget: child's total must fit within parent's remaining.
        if child.risk_budget.total > self.risk_budget.remaining:
            return (
                False,
                f"child.risk_budget.total={child.risk_budget.total} exceeds "
                f"parent.risk_budget.remaining={self.risk_budget.remaining}",
            )
        return (True, "ok")


# ---------------------------------------------------------------------------
# Delegation Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelegationContract:
    """A signed-or-should-have-been-signed token of authority.

    Tracks:
    - WHO delegated WHAT to WHOM (delegator → delegatee)
    - In service of WHICH goal (goal_id)
    - Within WHAT scope (allowed_actions, allowed_surfaces)
    - Under WHAT budget (ContractBudget)
    - With WHAT lifetime (issued_at, expires_at)
    - As PART OF which authority chain (root_intent_id, parent_contract_id)
    - At WHAT delegation depth (max_depth: 0 = leaf)
    """

    # Identity
    contract_id: str
    schema_version: str

    # Authority chain
    root_intent_id: str
    parent_contract_id: str | None
    delegator: str
    delegatee: str
    max_depth: int

    # Goal binding
    goal_id: str

    # Scope (ocap dimension)
    allowed_actions: frozenset[str]
    denied_actions: frozenset[str]
    allowed_surfaces: AllowedSurfaces
    destructive_action_policy: DestructivePolicy

    # Budget (gas dimension)
    budget: ContractBudget

    # Lifecycle
    issued_at: str
    expires_at: str
    revocation_check_uri: str | None

    # Progress gating
    progress_predicates: list[str]
    stale_threshold_minutes: int

    # v0.2 stub for v0.4 signing
    signature: str | None = None

    # -----------------------------------------------------------------------
    # Self-validation
    # -----------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ContractValidationError if this contract is self-inconsistent."""
        if not self.contract_id:
            raise ContractValidationError("contract_id must not be empty")
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"unexpected schema_version: {self.schema_version!r}; expected "
                f"{CONTRACT_SCHEMA_VERSION!r}"
            )
        if not self.root_intent_id:
            raise ContractValidationError("root_intent_id must not be empty")
        if not self.delegator or not self.delegatee:
            raise ContractValidationError("delegator and delegatee must not be empty")
        if self.parent_contract_id is None and self.delegator != self.delegatee:
            # Root contracts: human delegates to themselves OR to an agent;
            # both are valid as long as the human signed it. v0.1 has no
            # signing, so we only enforce non-empty.
            pass
        if self.max_depth < 0:
            raise ContractValidationError(f"max_depth must be >= 0; got {self.max_depth}")
        if not self.goal_id:
            raise ContractValidationError("goal_id must not be empty")
        # Time fields
        _parse_utc(self.issued_at, "issued_at")
        expires = _parse_utc(self.expires_at, "expires_at")
        if expires <= _parse_utc(self.issued_at, "issued_at"):
            raise ContractValidationError(
                f"expires_at ({self.expires_at}) must be after issued_at ({self.issued_at})"
            )
        # Destructive policy in v0.1: cannot be "allow" — destructive actions
        # always require a human gate that v0.1 doesn't carry.
        if self.destructive_action_policy == "allow":
            raise ContractValidationError(
                "destructive_action_policy='allow' is not permitted in v0.1; "
                "use 'deny' or 'human-only'"
            )
        # Destructive actions cannot appear in allowed_actions unless policy is
        # "human-only" (in v0.1 the carrier for the human-approval-token
        # doesn't exist, so effectively destructive actions always go through
        # an out-of-band approval gate).
        for action in self.allowed_actions:
            if _is_destructive_action(action) and self.destructive_action_policy == "deny":
                raise ContractValidationError(
                    f"contract allows destructive action {action!r} but "
                    f"destructive_action_policy='deny'"
                )
        # Budget sanity
        self.budget.validate()
        # Stale threshold
        if self.stale_threshold_minutes < 1:
            raise ContractValidationError(
                f"stale_threshold_minutes must be >= 1; got {self.stale_threshold_minutes}"
            )
        # Signature must be None in v0.1
        if self.signature is not None:
            raise ContractValidationError("signature must be None in v0.1; signing ships in v0.4")

    # -----------------------------------------------------------------------
    # Time helpers
    # -----------------------------------------------------------------------

    def is_expired(self, *, now: datetime | None = None) -> bool:
        moment = now if now is not None else datetime.now(UTC)
        return _parse_utc(self.expires_at, "expires_at") <= moment

    # -----------------------------------------------------------------------
    # Action filter
    # -----------------------------------------------------------------------

    def permits_action(self, action: str) -> tuple[bool, str]:
        """Whether this contract permits the given action string.

        Returns (allowed, reason).

        Order of operations:
        1. Destructive action against destructive_action_policy=='deny'
           → deny
        2. Explicit denied_actions match → deny
        3. allowed_actions match (or empty == any) → allow
        4. else → deny
        """
        if _is_destructive_action(action) and self.destructive_action_policy == "deny":
            return (False, f"action {action!r} is destructive and policy is 'deny'")
        # destructive_action_policy == "human-only" means the action is allowed
        # only if a human-approval-token is presented; v0.1 has no carrier so
        # effectively we deny here too. The caller (lane registry / wake_agent)
        # is responsible for surfacing the human-approval gate.
        if _is_destructive_action(action) and self.destructive_action_policy == "human-only":
            return (
                False,
                f"action {action!r} requires human approval; v0.1 has no carrier",
            )
        for pattern in self.denied_actions:
            if _action_matches(action, pattern):
                return (False, f"action {action!r} matches denied pattern {pattern!r}")
        if not self.allowed_actions:
            # Empty allowed set means "no positive grant"; v0.1 chooses deny-by-default.
            return (False, "no allowed_actions match (deny by default)")
        for pattern in self.allowed_actions:
            if _action_matches(action, pattern):
                return (True, f"matched allowed pattern {pattern!r}")
        return (False, f"action {action!r} matches no allowed pattern")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_utc(value: str, field_name: str) -> datetime:
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value[:-1] + "+00:00")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} is not valid ISO-8601 UTC: {value!r}") from exc


def _action_matches(action: str, pattern: str) -> bool:
    """fnmatch-style action matching. ':' is a literal separator; '*' is glob."""
    return fnmatch.fnmatchcase(action, pattern)


def _is_destructive_action(action: str) -> bool:
    return any(_action_matches(action, pattern) for pattern in DEFAULT_DESTRUCTIVE_ACTIONS)


# ---------------------------------------------------------------------------
# Monotonic narrowing
# ---------------------------------------------------------------------------


def narrow_for_child(
    parent: DelegationContract,
    *,
    child_contract_id: str,
    child_delegatee: str,
    child_allowed_actions: Iterable[str] | None = None,
    child_denied_actions: Iterable[str] | None = None,
    child_allowed_surfaces: AllowedSurfaces | None = None,
    child_budget: ContractBudget | None = None,
    child_expires_at: str | None = None,
    child_destructive_policy: DestructivePolicy | None = None,
    child_progress_predicates: list[str] | None = None,
    child_stale_threshold_minutes: int | None = None,
    issued_at: str | None = None,
) -> DelegationContract:
    """Issue a child contract narrowed from ``parent``.

    Enforces monotonic narrowing rules from the v0.1 spec:
    - child.goal_id == parent.goal_id
    - child.max_depth == parent.max_depth - 1
    - child.allowed_actions ⊆ parent.allowed_actions
    - child.denied_actions ⊇ parent.denied_actions
    - allowed_surfaces narrowed (or unchanged)
    - destructive policy cannot widen
    - budget ≤ parent.budget (each dimension, see ContractBudget.child_fits)
    - expires_at ≤ parent.expires_at
    - stale_threshold_minutes ≤ parent.stale_threshold_minutes

    Any violation raises ContractValidationError. The caller is responsible
    for actually persisting the child contract; this function only
    constructs and validates it.

    Defaults for omitted child fields inherit from parent (mostly — the
    monotonic-narrowing semantic).
    """
    parent.validate()

    if parent.max_depth <= 0:
        raise ContractValidationError(
            f"parent contract has max_depth={parent.max_depth}; cannot spawn children"
        )

    issued = issued_at if issued_at is not None else _now_utc_iso()

    # Inherit unchanged where caller didn't override
    allowed_actions = (
        frozenset(child_allowed_actions)
        if child_allowed_actions is not None
        else parent.allowed_actions
    )
    denied_actions = (
        frozenset(child_denied_actions)
        if child_denied_actions is not None
        else parent.denied_actions
    )
    allowed_surfaces = (
        child_allowed_surfaces if child_allowed_surfaces is not None else parent.allowed_surfaces
    )
    budget = child_budget if child_budget is not None else parent.budget
    expires_at = child_expires_at if child_expires_at is not None else parent.expires_at
    destructive_policy = (
        child_destructive_policy
        if child_destructive_policy is not None
        else parent.destructive_action_policy
    )
    progress_predicates = (
        list(child_progress_predicates)
        if child_progress_predicates is not None
        else list(parent.progress_predicates)
    )
    stale_threshold = (
        child_stale_threshold_minutes
        if child_stale_threshold_minutes is not None
        else parent.stale_threshold_minutes
    )

    # --- Enforce narrowing rules ---

    # Actions: child ⊆ parent
    extra_allowed = allowed_actions - parent.allowed_actions
    if extra_allowed:
        raise ContractValidationError(
            f"child allowed_actions widen parent's: extra={sorted(extra_allowed)}"
        )
    # Denied: child ⊇ parent
    missing_denied = parent.denied_actions - denied_actions
    if missing_denied:
        raise ContractValidationError(
            f"child denied_actions narrow parent's: missing={sorted(missing_denied)}"
        )

    # Surfaces
    _check_surfaces_narrowed(parent.allowed_surfaces, allowed_surfaces)

    # Destructive policy: cannot widen
    if destructive_policy == "allow" and parent.destructive_action_policy != "allow":
        raise ContractValidationError("child destructive_action_policy widens parent's")
    if destructive_policy == "human-only" and parent.destructive_action_policy == "deny":
        raise ContractValidationError(
            "child destructive_action_policy widens parent's (parent='deny', child='human-only')"
        )

    # Budget
    fits, reason = parent.budget.child_fits(budget)
    if not fits:
        raise ContractValidationError(f"child budget exceeds parent: {reason}")

    # Time
    parent_expires = _parse_utc(parent.expires_at, "parent.expires_at")
    child_expires_dt = _parse_utc(expires_at, "child.expires_at")
    if child_expires_dt > parent_expires:
        raise ContractValidationError(
            f"child.expires_at ({expires_at}) is after parent.expires_at ({parent.expires_at})"
        )

    # Stale threshold (children fail faster)
    if stale_threshold > parent.stale_threshold_minutes:
        raise ContractValidationError(
            f"child.stale_threshold_minutes={stale_threshold} exceeds "
            f"parent.stale_threshold_minutes={parent.stale_threshold_minutes}"
        )

    # Issued-time sanity
    issued_dt = _parse_utc(issued, "issued_at")
    if issued_dt > child_expires_dt:
        raise ContractValidationError(
            f"issued_at ({issued}) is after child.expires_at ({expires_at})"
        )

    child = DelegationContract(
        contract_id=child_contract_id,
        schema_version=CONTRACT_SCHEMA_VERSION,
        root_intent_id=parent.root_intent_id,
        parent_contract_id=parent.contract_id,
        delegator=parent.delegatee,
        delegatee=child_delegatee,
        max_depth=parent.max_depth - 1,
        goal_id=parent.goal_id,
        allowed_actions=allowed_actions,
        denied_actions=denied_actions,
        allowed_surfaces=allowed_surfaces,
        destructive_action_policy=destructive_policy,
        budget=budget,
        issued_at=issued,
        expires_at=expires_at,
        revocation_check_uri=parent.revocation_check_uri,
        progress_predicates=progress_predicates,
        stale_threshold_minutes=stale_threshold,
        signature=None,
    )
    child.validate()
    return child


def _check_surfaces_narrowed(parent: AllowedSurfaces, child: AllowedSurfaces) -> None:
    """Each child surface set must be a subset of (or contained within)
    the parent's, except for ``deny_file_globs`` which must be a superset."""
    # PR numbers: empty parent means any; otherwise child ⊆ parent
    if parent.pr_numbers and (child.pr_numbers - parent.pr_numbers):
        raise ContractValidationError(
            f"child allowed_surfaces.pr_numbers widens parent's: extra="
            f"{sorted(child.pr_numbers - parent.pr_numbers)}"
        )
    # Globs: every child glob must be contained by at least one parent glob.
    for kind, child_set, parent_set in (
        ("branch_globs", child.branch_globs, parent.branch_globs),
        ("worktree_globs", child.worktree_globs, parent.worktree_globs),
        ("file_globs", child.file_globs, parent.file_globs),
    ):
        if not parent_set:
            continue  # parent has no restriction, child may add any
        for cg in child_set:
            if not any(_glob_contains(pg, cg) for pg in parent_set):
                raise ContractValidationError(
                    f"child allowed_surfaces.{kind} glob {cg!r} is not contained "
                    f"by any parent glob in {sorted(parent_set)}"
                )
    # Deny file globs: child ⊇ parent (child denies at least everything parent denies)
    missing = parent.deny_file_globs - child.deny_file_globs
    if missing:
        raise ContractValidationError(
            f"child allowed_surfaces.deny_file_globs is missing parent denies: {sorted(missing)}"
        )


def _glob_contains(parent_glob: str, child_glob: str) -> bool:
    """Heuristic: does parent_glob's set ⊇ child_glob's set?

    Exact-match is sufficient. For wildcards, we accept the conservative
    rule: parent ends with '*' AND child starts with parent's stem, OR
    parent == child.

    This is intentionally simple in v0.1; the test suite documents the
    accepted shapes. v0.2+ can swap in a full glob-containment algorithm
    if needed.
    """
    if parent_glob == child_glob:
        return True
    if parent_glob == "*":
        return True
    if parent_glob.endswith("/*"):
        prefix = parent_glob[:-2]
        if child_glob == prefix:
            return True
        if child_glob.startswith(prefix + "/"):
            return True
        # child_glob is itself a glob — accept if it's strictly under prefix
        if child_glob.startswith(prefix + "/") and child_glob.endswith("*"):
            return True
    if parent_glob.endswith("*"):
        prefix = parent_glob[:-1]
        return child_glob.startswith(prefix)
    return False


def _now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Convenience builder for tests / examples
# ---------------------------------------------------------------------------


def make_root_contract(
    *,
    contract_id: str,
    root_intent_id: str,
    delegator: str,
    delegatee: str,
    goal_id: str,
    allowed_actions: Iterable[str] = ("read:*",),
    denied_actions: Iterable[str] = (),
    allowed_surfaces: AllowedSurfaces | None = None,
    budget: ContractBudget | None = None,
    max_depth: int = 3,
    duration_minutes: int = 60,
    stale_threshold_minutes: int = 30,
    destructive_action_policy: DestructivePolicy = "deny",
    progress_predicates: list[str] | None = None,
    revocation_check_uri: str | None = None,
) -> DelegationContract:
    """Build and validate a root contract. Convenience for tests + example
    docs; production callers should use a signed-issuance flow when v0.4
    lands.
    """
    issued = _now_utc_iso()
    expires_at = (datetime.now(UTC) + timedelta(minutes=duration_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    surfaces = allowed_surfaces if allowed_surfaces is not None else AllowedSurfaces()
    contract_budget = budget if budget is not None else ContractBudget(risk_budget=RiskBudget())
    contract = DelegationContract(
        contract_id=contract_id,
        schema_version=CONTRACT_SCHEMA_VERSION,
        root_intent_id=root_intent_id,
        parent_contract_id=None,
        delegator=delegator,
        delegatee=delegatee,
        max_depth=max_depth,
        goal_id=goal_id,
        allowed_actions=frozenset(allowed_actions),
        denied_actions=frozenset(denied_actions),
        allowed_surfaces=surfaces,
        destructive_action_policy=destructive_action_policy,
        budget=contract_budget,
        issued_at=issued,
        expires_at=expires_at,
        revocation_check_uri=revocation_check_uri,
        progress_predicates=list(progress_predicates or []),
        stale_threshold_minutes=stale_threshold_minutes,
        signature=None,
    )
    contract.validate()
    return contract
