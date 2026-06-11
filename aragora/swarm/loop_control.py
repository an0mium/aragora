"""Loop Control Plane v1 - pure classification (no IO).

Read-only governance observability for Aragora's standing loops (boss loop,
merge arbiter, proof-first shift, publisher, worktree autopilot, nomic,
docs-sync drift detector). This
module is *pure*: it takes already-collected raw signals (see
``loop_control_io``) plus a static per-loop spec and returns normalized
``LoopRecord`` objects. It performs no subprocess, filesystem, or network IO,
which keeps it trivially unit-testable and keeps the read-only guarantee
auditable - only the IO layer touches the world.

It applies the loop-governance lesson that a loop is only safe to keep running
when it (1) is bounded by a max iteration/runtime, (2) has no-progress detection
that distinguishes a genuine *operational fault* (halt) from *normal waiting*
(continue), and (3) has a spend ceiling. See
``docs/governance/LOOP_CONTROL_PLANE.md``.

The halt-readiness guards in ``LOOP_SPECS`` are a *curated* design-level audit
with code references, not auto-derived facts; update them when a loop's guards
change. v1 deliberately does not auto-derive them (see doc follow-ups).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# Stable contract version. Carried on every record as a forward-compatible hook
# for the future append-only LoopLedger (documented as a follow-up; not written
# in v1).
SCHEMA_VERSION = "loop-control/v1"


class LoopKind(str, Enum):
    BOSS_LOOP = "boss_loop"
    MERGE_ARBITER = "merge_arbiter"
    PROOF_FIRST_SHIFT = "proof_first_shift"
    PUBLISHER = "publisher"
    WORKTREE_AUTOPILOT = "worktree_autopilot"
    NOMIC = "nomic"
    DOCS_SYNC_DRIFT = "docs_sync_drift"


class LoopState(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    HALTED = "halted"
    HUMAN_GATED = "human_gated"
    STALE_OWNER = "stale_owner"
    UNKNOWN = "unknown"


class NextAction(str, Enum):
    REPORT_ONLY = "report_only"
    CONTINUE = "continue"
    WAIT = "wait"
    HALT = "halt"
    ESCALATE_HUMAN = "escalate_human"


class HaltVerdict(str, Enum):
    OK = "ok"
    INCOMPLETE = "incomplete"
    MISSING = "missing"


# Stop-reason fragments that denote a *normal* terminal state (not a fault).
# Anything else is treated as an operational fault (fail-closed).
_NORMAL_STOP_FRAGMENTS = (
    "max runtime reached",
    "max-runtime",
    "max runtime",
    "timelimit",
    "time limit",
    "no candidates",
    "no candidate",
    "completed",
    "deadline reached",
    "shift complete",
)


def _is_fault_stop(stop_reason: str) -> bool:
    """True when ``stop_reason`` denotes an operational fault that should halt.

    Fail-closed: an unrecognized non-empty stop reason is treated as a fault.
    """
    text = stop_reason.strip().lower()
    if not text:
        return False
    return not any(fragment in text for fragment in _NORMAL_STOP_FRAGMENTS)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class HaltGuards:
    """Curated, design-level declaration of a loop's halt guards.

    Reflects the loop's source as of ``code_ref``; update when guards change.
    """

    max_iteration: bool
    no_progress: bool
    no_progress_distinguishes_fault: bool
    budget_ceiling: bool
    code_ref: str = ""
    notes: tuple[str, ...] = ()


@dataclass
class HaltReadiness:
    max_iteration: bool
    no_progress: bool
    no_progress_distinguishes_fault: bool
    budget_ceiling: bool
    verdict: str
    gaps: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_halt_readiness(guards: HaltGuards) -> HaltReadiness:
    """Audit a loop's halt guards against the three hard stops.

    ``ok`` requires all three: a max iteration/runtime bound, no-progress
    detection that distinguishes operational fault from normal waiting, and a
    budget ceiling. Anything partial is ``incomplete``; none is ``missing``.
    """
    gaps: list[str] = []
    if not guards.max_iteration:
        gaps.append("no max-iteration/runtime bound")
    if not guards.no_progress:
        gaps.append("no no-progress detection")
    elif not guards.no_progress_distinguishes_fault:
        gaps.append(
            "no-progress detection does not distinguish operational fault from normal waiting"
        )
    if not guards.budget_ceiling:
        gaps.append("no dollar/budget ceiling (bounded by time/iterations only)")

    has_iteration = guards.max_iteration
    has_no_progress = guards.no_progress and guards.no_progress_distinguishes_fault
    has_budget = guards.budget_ceiling
    present = sum((has_iteration, has_no_progress, has_budget))
    if present == 3:
        verdict = HaltVerdict.OK.value
    elif present == 0:
        verdict = HaltVerdict.MISSING.value
    else:
        verdict = HaltVerdict.INCOMPLETE.value

    return HaltReadiness(
        max_iteration=guards.max_iteration,
        no_progress=guards.no_progress,
        no_progress_distinguishes_fault=guards.no_progress_distinguishes_fault,
        budget_ceiling=guards.budget_ceiling,
        verdict=verdict,
        gaps=gaps,
        notes=list(guards.notes),
    )


@dataclass(frozen=True)
class LoopSpec:
    kind: LoopKind
    title: str
    role: str  # supervisor | orchestration | maintenance | publication | self_improvement
    guards: HaltGuards
    feedback_kind: (
        str  # quorum | proof_freshness | publisher_freshness | worktree_health | docs_drift | none
    )
    durable_state_path: str | None
    human_gate_required: bool
    source_paths: tuple[str, ...]


@dataclass
class Budget:
    spend_usd: float | None
    ceiling_usd: float | None
    remaining_usd: float | None
    source: str
    source_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackGate:
    kind: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Durability:
    state_path: str | None
    restart_safe: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HumanGate:
    required: bool
    present: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopRecord:
    schema_version: str
    loop_id: str
    kind: str
    role: str
    owner: str | None
    state: str
    ticks: int | None
    max_ticks: int | None
    runtime_s: float | None
    max_runtime_s: float | None
    last_progress_at: str | None
    no_progress_ticks: int | None
    budget: Budget
    feedback_gate: FeedbackGate
    halt_readiness: HaltReadiness
    durability: Durability
    human_gate: HumanGate
    blocker: str | None
    next_action: str
    source_paths: list[str] = field(default_factory=list)
    source_status: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_loop(spec: LoopSpec, raw: dict[str, Any]) -> LoopRecord:
    """Classify one loop's normalized raw signals into a ``LoopRecord``.

    Pure and fail-closed: missing/unknown signals yield ``state=unknown`` with
    ``next_action=report_only``; an operational fault or exhausted budget yields
    ``halt``; a not-ready *waiting* loop yields ``wait`` (continue waiting), never
    a halt - the distinction that the merge-arbiter circuit-breaker bug (#7879)
    got wrong.
    """
    source_status = str(raw.get("source_status", "unavailable"))
    owner = raw.get("owner")
    alive = raw.get("alive")
    stop_reason = raw.get("stop_reason")
    operational_fault = bool(raw.get("operational_fault", False))
    waiting_only = bool(raw.get("waiting_only", False))
    owner_stale = bool(raw.get("owner_stale", False))
    awaiting_human = bool(raw.get("awaiting_human", False))
    human_present = bool(raw.get("human_settlement_present", False))

    raw_budget_value = raw.get("budget")
    raw_budget: dict[str, Any] = raw_budget_value if isinstance(raw_budget_value, dict) else {}
    budget = Budget(
        spend_usd=_as_float(raw_budget.get("spend_usd")),
        ceiling_usd=_as_float(raw_budget.get("ceiling_usd")),
        remaining_usd=_as_float(raw_budget.get("remaining_usd")),
        source=str(raw_budget.get("source", "none")),
        source_status=str(raw_budget.get("source_status", "unavailable")),
    )

    halt_readiness = audit_halt_readiness(spec.guards)
    feedback_gate = FeedbackGate(
        kind=spec.feedback_kind,
        status=str(raw.get("feedback_status", "unknown")),
    )
    human_gate = HumanGate(required=spec.human_gate_required, present=human_present)
    durability = Durability(
        state_path=spec.durable_state_path,
        restart_safe=spec.durable_state_path is not None,
    )

    blocker: str | None = None
    has_live_signal = alive is not None or bool(stop_reason) or operational_fault

    if source_status == "unavailable" and not has_live_signal:
        state, action = LoopState.UNKNOWN, NextAction.REPORT_ONLY
    elif spec.human_gate_required and awaiting_human and not human_present:
        state, action = LoopState.HUMAN_GATED, NextAction.ESCALATE_HUMAN
        blocker = "awaiting human settlement"
    elif budget.remaining_usd is not None and budget.remaining_usd <= 0:
        state, action = LoopState.BUDGET_EXHAUSTED, NextAction.HALT
        blocker = "budget exhausted"
    elif operational_fault or (isinstance(stop_reason, str) and _is_fault_stop(stop_reason)):
        state, action = LoopState.BLOCKED, NextAction.HALT
        blocker = (
            stop_reason if isinstance(stop_reason, str) and stop_reason else "operational fault"
        )
    elif owner_stale:
        state, action = LoopState.STALE_OWNER, NextAction.REPORT_ONLY
        blocker = "owner heartbeat stale"
    elif alive is True and waiting_only:
        state, action = LoopState.WAITING, NextAction.WAIT
    elif alive is True:
        state, action = LoopState.RUNNING, NextAction.CONTINUE
    elif alive is False:
        # Cleanly stopped with a normal (non-fault) reason, or simply idle.
        state, action = LoopState.HALTED, NextAction.REPORT_ONLY
        blocker = stop_reason if isinstance(stop_reason, str) and stop_reason else None
    else:
        state, action = LoopState.UNKNOWN, NextAction.REPORT_ONLY

    return LoopRecord(
        schema_version=SCHEMA_VERSION,
        loop_id=spec.kind.value,
        kind=spec.kind.value,
        role=spec.role,
        owner=owner if isinstance(owner, str) else None,
        state=state.value,
        ticks=_as_int(raw.get("ticks")),
        max_ticks=_as_int(raw.get("max_ticks")),
        runtime_s=_as_float(raw.get("runtime_s")),
        max_runtime_s=_as_float(raw.get("max_runtime_s")),
        last_progress_at=(
            raw.get("last_progress_at") if isinstance(raw.get("last_progress_at"), str) else None
        ),
        no_progress_ticks=_as_int(raw.get("no_progress_ticks")),
        budget=budget,
        feedback_gate=feedback_gate,
        halt_readiness=halt_readiness,
        durability=durability,
        human_gate=human_gate,
        blocker=blocker,
        next_action=action.value,
        source_paths=list(spec.source_paths),
        source_status=source_status,
    )


def summarize(records: list[LoopRecord]) -> dict[str, Any]:
    """Fleet-level rollup of loop records (pure)."""
    by_state: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_halt: dict[str, int] = {}
    for record in records:
        by_state[record.state] = by_state.get(record.state, 0) + 1
        by_action[record.next_action] = by_action.get(record.next_action, 0) + 1
        verdict = record.halt_readiness.verdict
        by_halt[verdict] = by_halt.get(verdict, 0) + 1

    any_blocked = any(
        r.state in (LoopState.BLOCKED.value, LoopState.BUDGET_EXHAUSTED.value) for r in records
    )
    any_human_gated = any(r.state == LoopState.HUMAN_GATED.value for r in records)
    halt_gaps = [
        {"loop": r.kind, "verdict": r.halt_readiness.verdict, "gaps": r.halt_readiness.gaps}
        for r in records
        if r.halt_readiness.verdict != HaltVerdict.OK.value
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "loops": len(records),
        "by_state": by_state,
        "by_next_action": by_action,
        "by_halt_verdict": by_halt,
        "any_blocked": any_blocked,
        "any_human_gated": any_human_gated,
        "fleet_safe_to_continue": not any_blocked and not any_human_gated,
        "halt_readiness_gaps": halt_gaps,
    }


# ---------------------------------------------------------------------------
# Static loop registry (curated halt-readiness audit; see module docstring).
# Guard values reflect current ``origin/main`` with the cited code references.
# ---------------------------------------------------------------------------
LOOP_SPECS: dict[LoopKind, LoopSpec] = {
    LoopKind.BOSS_LOOP: LoopSpec(
        kind=LoopKind.BOSS_LOOP,
        title="Boss loop (settlement supervisor)",
        role="supervisor",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=True,
            no_progress_distinguishes_fault=True,
            budget_ceiling=False,
            code_ref="scripts/run_boss_cycle.sh -> aragora/swarm/boss_loop.py",
            notes=(
                "self-heal/unstick recovery is advisory, not executed deterministically "
                "(BOSS_LOOP_MERGE_GATE_RESILIENCE.md root cause #5)",
            ),
        ),
        feedback_kind="quorum",
        durable_state_path=".aragora/operator_steering",
        human_gate_required=False,
        source_paths=("scripts/run_boss_cycle.sh", "aragora/swarm/boss_loop.py"),
    ),
    LoopKind.MERGE_ARBITER: LoopSpec(
        kind=LoopKind.MERGE_ARBITER,
        title="Merge arbiter (auto-merge-on-green)",
        role="supervisor",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=True,
            no_progress_distinguishes_fault=True,
            budget_ceiling=False,
            code_ref=(
                "aragora/swarm/merge_arbiter.py ArbiterOperationalError + MergeArbiter.run "
                "(max_runtime_hours=12, max_consecutive_failures=3)"
            ),
            notes=(
                "breaker trips only on systemic operational faults: a candidate list-fetch "
                "fault, or every evaluation in a poll faulting; not-ready PRs and a single "
                "poison-pill PR never trip it (#7879, PR #8125). Residual: merge-API failures "
                "during a merge attempt are recorded as results, not faults, and are bounded "
                "by max_runtime_hours",
            ),
        ),
        feedback_kind="quorum",
        durable_state_path=None,
        human_gate_required=True,
        source_paths=("scripts/run_merge_arbiter.sh", "aragora/swarm/merge_arbiter.py"),
    ),
    LoopKind.PROOF_FIRST_SHIFT: LoopSpec(
        kind=LoopKind.PROOF_FIRST_SHIFT,
        title="Proof-first shift (bounded Foreman)",
        role="orchestration",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=True,
            no_progress_distinguishes_fault=True,
            budget_ceiling=False,
            code_ref="scripts/run_proof_first_shift.py RECOVERY_STOP_REASONS + AUTH_FAILURE_STOP_AFTER",
            notes=(
                "recovery failures are classified by type and fail closed after bounded retries",
            ),
        ),
        feedback_kind="proof_freshness",
        durable_state_path=".aragora/proof_first_shift",
        human_gate_required=False,
        source_paths=("scripts/run_proof_first_shift.py",),
    ),
    LoopKind.PUBLISHER: LoopSpec(
        kind=LoopKind.PUBLISHER,
        title="Codex automation publisher",
        role="publication",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=True,
            no_progress_distinguishes_fault=True,
            budget_ceiling=False,
            code_ref="scripts/publisher_freshness_check.py (launchd single-shot per fire)",
            notes=("launchd cron single-shot; freshness verdict separates warming from degraded",),
        ),
        feedback_kind="publisher_freshness",
        durable_state_path=".aragora/automation-github-status",
        human_gate_required=False,
        source_paths=(
            "scripts/run_codex_automation_publisher.sh",
            "scripts/publisher_freshness_check.py",
        ),
    ),
    LoopKind.WORKTREE_AUTOPILOT: LoopSpec(
        kind=LoopKind.WORKTREE_AUTOPILOT,
        title="Worktree autopilot (reconcile/cleanup)",
        role="maintenance",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=False,
            no_progress_distinguishes_fault=False,
            budget_ceiling=False,
            code_ref="scripts/codex_worktree_autopilot.py + scripts/worktree_maintainer.sh (TTL-bounded)",
            notes=("TTL-bounded maintenance reconcile; no progress metric applies",),
        ),
        feedback_kind="worktree_health",
        durable_state_path=".worktrees",
        human_gate_required=False,
        source_paths=(
            "scripts/codex_worktree_autopilot.py",
            "scripts/worktree_maintainer.sh",
        ),
    ),
    LoopKind.NOMIC: LoopSpec(
        kind=LoopKind.NOMIC,
        title="Nomic self-improvement loop",
        role="self_improvement",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=False,
            no_progress_distinguishes_fault=False,
            budget_ceiling=False,
            code_ref="scripts/nomic_loop.py (--cycles bound; protected file)",
            notes=("cycle-bounded with human approval checkpoints; protected file",),
        ),
        feedback_kind="none",
        durable_state_path=".aragora_beads",
        human_gate_required=True,
        source_paths=("scripts/nomic_loop.py",),
    ),
    LoopKind.DOCS_SYNC_DRIFT: LoopSpec(
        kind=LoopKind.DOCS_SYNC_DRIFT,
        title="Docs-site sync drift detector",
        role="maintenance",
        guards=HaltGuards(
            max_iteration=True,
            no_progress=True,
            no_progress_distinguishes_fault=True,
            budget_ceiling=False,
            code_ref=(
                "scripts/docs_sync_drift_detector.py (launchd single-shot per fire; "
                "FAULT_OUTCOMES separates fault from waiting-on-PR; fail-closed outside "
                "the generated-mirror allowlist)"
            ),
            notes=(
                "sync PRs settle through the normal model-quorum gate; "
                "the detector never merges, approves, or comments",
            ),
        ),
        feedback_kind="docs_drift",
        durable_state_path=".aragora/docs_drift_status.json",
        human_gate_required=False,
        source_paths=(
            "scripts/docs_sync_drift_detector.py",
            "scripts/install_docs_drift_launchd.sh",
        ),
    ),
}
