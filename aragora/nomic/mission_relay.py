"""Relay-with-timeout decision core for the native mission orchestrator.

The piece that stops a blocked item from halting the whole mission (and the
operator from becoming a 3-minute relay). Pure and deterministic: given a
policy + the observed state of one work item, it decides whether to keep going,
**park the item and notify** (so the mission continues OTHER items), or stop the
whole mission. No I/O, no notifications sent here — the actual Slack/email send
and the boss_loop wiring land in a separate quiet-window PR (boss_loop.py drives
the live loop and must not be edited while it is running).

Hard invariant: the only outcomes are CONTINUE, PARK_AND_NOTIFY, or STOP_MISSION.
There is NO outcome that merges, settles, or marks-ready anything — parking sets
an item aside for the operator; the merge-quorum gate stays the sole authority.
STOP_MISSION is reserved for genuine global limits (budget/time), never for a
single blocked item.

PR #3 of docs/plans/2026-06-16-native-mission-orchestrator.md (pure core only).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RelayAction(Enum):
    CONTINUE = "continue"  # item healthy / transient — keep working it
    PARK_AND_NOTIFY = (
        "park_and_notify"  # set item aside + notify operator; mission continues others
    )
    STOP_MISSION = "stop_mission"  # global limit hit (budget/time) — wind the whole mission down


@dataclass(frozen=True)
class RelayPolicy:
    """How tolerant the mission is before parking an item or stopping."""

    max_item_failures: int = 3  # consecutive failures on one item -> park it
    item_timeout_seconds: float | None = None  # per-item wall-clock cap -> park (None = no cap)
    relay_channel: str = "none"  # where PARK notifications go: none|slack|email

    def __post_init__(self) -> None:
        if self.max_item_failures < 1:
            raise ValueError("max_item_failures must be >= 1")
        if self.item_timeout_seconds is not None and self.item_timeout_seconds <= 0:
            raise ValueError("item_timeout_seconds must be positive or None")
        if self.relay_channel not in {"none", "slack", "email"}:
            raise ValueError("relay_channel must be one of none|slack|email")


@dataclass(frozen=True)
class RelayContext:
    """Observed state for one work item at a decision point."""

    item_id: str
    consecutive_failures: int = 0
    needs_human: bool = False
    elapsed_seconds: float = 0.0
    budget_exhausted: bool = False
    max_hours_exceeded: bool = False


@dataclass(frozen=True)
class RelayDecision:
    action: RelayAction
    reason: str
    notify: bool  # emit a relay notification (only meaningful when channel != "none")

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, "reason": self.reason, "notify": self.notify}


def decide_relay_action(policy: RelayPolicy, ctx: RelayContext) -> RelayDecision:
    """Decide the relay action for one item. Pure; global limits take precedence.

    Order: (1) global budget/time limits -> STOP_MISSION; (2) per-item
    needs-human / repeated-failure / timeout -> PARK_AND_NOTIFY (mission keeps
    running other items); (3) otherwise CONTINUE. A PARK notifies only when a
    real channel is configured.
    """
    # (1) Global limits stop the whole mission — these are the only STOP causes.
    if ctx.budget_exhausted:
        return RelayDecision(RelayAction.STOP_MISSION, "budget exhausted", notify=True)
    if ctx.max_hours_exceeded:
        return RelayDecision(RelayAction.STOP_MISSION, "max_hours exceeded", notify=True)

    # (2) Per-item blocks PARK the item (never STOP the mission, never merge/settle).
    notify = policy.relay_channel != "none"
    if ctx.needs_human:
        return RelayDecision(RelayAction.PARK_AND_NOTIFY, "item needs human", notify=notify)
    if ctx.consecutive_failures >= policy.max_item_failures:
        return RelayDecision(
            RelayAction.PARK_AND_NOTIFY,
            f"item hit {ctx.consecutive_failures} consecutive failures (>= {policy.max_item_failures})",
            notify=notify,
        )
    if (
        policy.item_timeout_seconds is not None
        and ctx.elapsed_seconds >= policy.item_timeout_seconds
    ):
        return RelayDecision(
            RelayAction.PARK_AND_NOTIFY,
            f"item exceeded timeout {policy.item_timeout_seconds}s",
            notify=notify,
        )

    # (3) Healthy / transient — keep working it.
    return RelayDecision(RelayAction.CONTINUE, "item healthy", notify=False)
