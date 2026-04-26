"""PR intelligence brief — cost-aware policy + budget controls (#6305 foundation).

Type contracts for the review-depth policy and review budget described in
docs/plans/2026-04-19-pr-intelligence-brief.md (#6305).

This module is **schema only**. No I/O, no evaluation, no metering.
Behavior — applying policy to a PR, reading cost tracker state, denying
a run that exceeds budget — ships in the #6306 orchestrator + review-queue
writer. This module just defines what configuring and reporting on policy
looks like.

Composes with, does not duplicate, existing billing infrastructure:
  - ``aragora.billing.budget_policy.BudgetPolicy`` is the generic
    workspace-level budget. ``ReviewBudget`` in this module is the
    PR-review-specific slice. A caller that already has a generic
    BudgetPolicy can derive a ReviewBudget from it; this module doesn't
    force re-entry.
  - ``aragora.policy.engine.PolicyDecision`` has its own
    ALLOW/DENY/ESCALATE/BUDGET_EXCEEDED vocabulary for deploy decisions.
    ``ReviewPolicyDecision`` here has a review-specific DEGRADE value
    (drop to a cheaper review depth rather than refuse entirely) that
    the generic enum does not express.

Safe-default posture (per #6305 acceptance "Safe defaults for Aragora
dogfood before wider rollout"): the defaults on this module are
deliberately conservative — a $25/PR cap, 80% alert threshold, and
STANDARD default depth. Tests lock those values so future wider-rollout
decisions require explicit rewrites, not silent drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# --- Enums ----------------------------------------------------------------


class ReviewDepth(str, Enum):
    """How deep a review run should go.

    Ordered from cheapest to most thorough. The runner (a successor PR,
    not this module) picks the panel / rounds / synthesis_policy
    appropriate for a given depth.
    """

    TRIVIAL = "trivial"  # one cheap model, single pass, formatting/typos only
    STANDARD = "standard"  # the default heterogeneous panel, single pass
    DEEP = "deep"  # multi-round panel with synthesizer + deeper evidence fetch


class RiskClass(str, Enum):
    """How much blast radius a PR carries.

    Distinct from ``aragora.policy.risk.RiskLevel`` (deployment blast
    radius) — this one is specifically about PR-review depth selection.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewPolicyDecision(str, Enum):
    """Outcome when a policy evaluates a candidate review run.

    ``DEGRADE`` is the review-specific value that the generic
    ``aragora.policy.engine.PolicyDecision`` does not express: drop to a
    cheaper depth rather than refuse entirely. Keeps coverage non-zero
    under budget pressure.
    """

    ALLOW = "allow"  # run at the selected depth
    DEGRADE = "degrade"  # run, but at a cheaper depth than requested
    DENY = "deny"  # refuse to run (budget exceeded or policy-forbidden)
    ESCALATE = "escalate"  # require human approval before running


class BudgetScope(str, Enum):
    """Which pool a budget cap / headroom line item applies to.

    Used by ``BudgetHeadroom`` and ``CostMeter.binding_scope`` so a
    packet-reader can tell which pool bound a DEGRADE/DENY decision.
    """

    PER_PR = "per_pr"
    PER_REPO_DAILY = "per_repo_daily"
    PER_ORG_DAILY = "per_org_daily"


# --- Depth rules ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DepthTrigger:
    """A single rule mapping a condition to a review depth.

    The first matching trigger in ``ReviewPolicy.depth_rules`` wins.
    A trigger matches when EVERY non-None / non-empty condition field
    matches the candidate PR; fields left at their default do not
    constrain the match.
    """

    target_depth: ReviewDepth
    # Diff size: if set, trigger matches when (additions + deletions) >= this.
    min_additions_plus_deletions: int = 0
    # Subsystems: if non-empty, trigger matches when any of these path
    # prefixes intersects the PR's touched-file list.
    subsystem_prefixes: tuple[str, ...] = ()
    # Risk: if set, trigger matches when the candidate PR's risk_class
    # is >= this one (using the enum declaration order as severity).
    min_risk_class: RiskClass | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_depth"] = self.target_depth.value
        d["subsystem_prefixes"] = list(self.subsystem_prefixes)
        if self.min_risk_class is not None:
            d["min_risk_class"] = self.min_risk_class.value
        return d


# --- Budget ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReviewBudget:
    """PR-review-specific spend caps.

    Composes with, does not replace, ``aragora.billing.budget_policy.BudgetPolicy``.
    The generic BudgetPolicy tracks monthly/daily/per-debate workspace spend;
    ReviewBudget carves out the review slice with per-PR and per-repo/org caps.

    Depth scoping (#6305 AC: "Per-repo / per-org caps for deep review
    runs"): daily repo/org pools only count runs at or above
    ``daily_caps_apply_at_or_above_depth``. This prevents a flood of
    TRIVIAL typo-only reviews from exhausting the daily pool and then
    denying a legitimate DEEP review later in the day. The per-PR cap
    has no depth scoping — it always applies to every single run.

    Defaults (per-PR $25, 80% alert threshold, daily caps apply at
    STANDARD+) are the dogfood-safe posture for Aragora's own PRs.
    Wider rollout requires an explicit override — tests lock the
    defaults so silent drift fails loudly.
    """

    per_pr_usd_cap: float = 25.0  # market anchor: Anthropic ~$25/PR; applies to ALL depths
    per_repo_usd_daily_cap: float = 0.0  # 0 = unlimited; depth-scoped
    per_org_usd_daily_cap: float = 0.0  # 0 = unlimited; depth-scoped
    daily_caps_apply_at_or_above_depth: ReviewDepth = ReviewDepth.STANDARD
    alert_threshold_pct: float = 80.0
    hard_limit: bool = True  # deny when cap reached; mirrors billing.BudgetPolicy

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["daily_caps_apply_at_or_above_depth"] = self.daily_caps_apply_at_or_above_depth.value
        return d


# --- Policy ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReviewPolicy:
    """Full review-depth + budget policy for a repo or org.

    Evaluation order (implementation lives in a successor PR, not here):
      1. Walk ``depth_rules`` in order; first matching trigger sets the
         chosen depth. If no trigger matches, use ``default_depth``.
      2. Estimate the cost of running at that depth.
      3. Apply ``budget`` caps: if estimate + spent-so-far would exceed
         the cap, emit DEGRADE (drop to a cheaper depth) or DENY
         (budget.hard_limit=True and no cheaper option).
      4. Emit a ReviewPolicyDecision plus the final depth.
    """

    budget: ReviewBudget = field(default_factory=ReviewBudget)
    depth_rules: tuple[DepthTrigger, ...] = ()
    default_depth: ReviewDepth = ReviewDepth.STANDARD

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["budget"] = self.budget.to_dict()
        d["depth_rules"] = [r.to_dict() for r in self.depth_rules]
        d["default_depth"] = self.default_depth.value
        return d


# --- Cost meter (what goes into the review packet per #6305 AC) -----------


@dataclass(frozen=True, slots=True)
class BudgetHeadroom:
    """Remaining/cap pair for one budget pool at decision time.

    Enables a packet-reader to answer "which pool was binding?" and
    "how much headroom did we have in each pool?" — not just a single
    anonymous remainder.
    """

    scope: BudgetScope
    cap_usd: float  # configured cap for this pool; 0.0 = unlimited
    remaining_usd: float  # headroom after this run; negative if over
    applies_at_or_above_depth: ReviewDepth | None = None  # None for per_pr (all depths)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scope"] = self.scope.value
        if self.applies_at_or_above_depth is not None:
            d["applies_at_or_above_depth"] = self.applies_at_or_above_depth.value
        return d


@dataclass(frozen=True, slots=True)
class CostMeter:
    """Cost-and-budget context embedded in a review packet.

    Addresses #6305 acceptance criterion: "Packet output includes cost
    used and budget context." Consumed by the UI (#6304) for the
    "packet cost:" line in the rendered brief, and by exporters so an
    operator reading an exported brief on another machine can still see
    whether the run was under budget AND which pool constrained it.

    Multi-pool disclosure (#6305 AC "bound deep-review spend without
    disabling the feature entirely"): when a DEGRADE or DENY decision
    is issued because a specific budget pool is exhausted, ``binding_scope``
    names which pool was binding and ``headroom_by_scope`` carries the
    per-pool remaining/cap tuple so the UI can explain to the operator
    exactly why the decision went the way it did.

    All cost fields are USD; units are implicit-by-convention because
    mixing currencies is out of scope for the foundation.
    """

    depth_chosen: ReviewDepth
    decision: ReviewPolicyDecision
    estimated_cost_usd: float  # what the runner estimated BEFORE running
    actual_cost_usd: float  # what the run actually spent (0 if pre-run)
    headroom_by_scope: tuple[BudgetHeadroom, ...]  # per-pool remaining+cap
    binding_scope: BudgetScope | None = None  # which pool bound the decision; None if unbounded
    alert_triggered: bool = False  # True if any pool's usage_pct crossed alert threshold

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["depth_chosen"] = self.depth_chosen.value
        d["decision"] = self.decision.value
        d["headroom_by_scope"] = [h.to_dict() for h in self.headroom_by_scope]
        if self.binding_scope is not None:
            d["binding_scope"] = self.binding_scope.value
        return d
