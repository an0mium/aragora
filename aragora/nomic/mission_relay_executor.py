"""Relay executor — the I/O half the pure decision core deferred.

`mission_relay.decide_relay_action` is pure: it only *decides* CONTINUE /
PARK_AND_NOTIFY / STOP_MISSION. This module turns that decision into real
effects: PARK_AND_NOTIFY marks the work item ``PARKED`` in the ``MissionStore``
and (when a relay channel is configured) sends an out-of-band notification;
STOP_MISSION sends a wind-down notification; CONTINUE is a no-op.

Hard invariant (inherited from the core): NOTHING here merges, settles, or
marks-ready anything. The only state change is ``WorkItemStatus -> PARKED`` plus
an advisory notification — the merge-quorum gate stays the sole authority, and a
parked item is set aside for the operator, never auto-resolved.

This is the executor half of PR #3 of
``docs/plans/2026-06-16-native-mission-orchestrator.md``. The boss_loop wiring
(calling ``evaluate_and_apply`` at the loop's per-item halt points) lands
separately, behind ``enable_native_mission``, in a quiet window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aragora.nomic.mission import MissionStore, WorkItemStatus
from aragora.nomic.mission_relay import (
    RelayAction,
    RelayContext,
    RelayDecision,
    RelayPolicy,
    decide_relay_action,
)
from aragora.notifications.models import Notification, NotificationChannel

# policy.relay_channel ("none"|"slack"|"email") -> concrete delivery channel.
_CHANNEL_MAP: dict[str, NotificationChannel] = {
    "slack": NotificationChannel.SLACK,
    "email": NotificationChannel.EMAIL,
}


class _Notifier(Protocol):
    """Minimal duck-typed surface of ``NotificationService`` the executor needs."""

    async def notify(
        self, notification: Notification, channels: list[NotificationChannel] | None = None
    ) -> Any: ...


@dataclass(frozen=True)
class RelayOutcome:
    """What the executor actually did for one item."""

    action: RelayAction
    reason: str
    parked: bool = False
    stopped: bool = False
    notified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "parked": self.parked,
            "stopped": self.stopped,
            "notified": self.notified,
        }


class MissionRelay:
    """Applies relay decisions: parks blocked items + notifies; never merges/settles."""

    def __init__(
        self,
        policy: RelayPolicy,
        store: MissionStore,
        *,
        notifier: _Notifier | None = None,
    ) -> None:
        self.policy = policy
        self.store = store
        self.notifier = notifier

    def evaluate(self, ctx: RelayContext) -> RelayDecision:
        """Pure decision (delegates to the core); no effects."""
        return decide_relay_action(self.policy, ctx)

    async def apply(self, mission_id: str, item_id: str, decision: RelayDecision) -> RelayOutcome:
        """Apply a decision: park the item (PARK) and/or notify; CONTINUE is a no-op.

        STOP_MISSION does NOT park the triggering item (it is a global limit, not
        an item fault) — it only signals the caller to wind the mission down.
        """
        if decision.action is RelayAction.PARK_AND_NOTIFY:
            parked = self.store.set_item_status(mission_id, item_id, WorkItemStatus.PARKED)
            # Only notify when the park actually took. If the item/mission vanished
            # (parked is False) there is nothing to set aside — sending a "parked"
            # alert would misinform the operator. The caller still sees parked=False.
            notified = await self._maybe_notify(mission_id, item_id, decision) if parked else False
            return RelayOutcome(decision.action, decision.reason, parked=parked, notified=notified)
        if decision.action is RelayAction.STOP_MISSION:
            notified = await self._maybe_notify(mission_id, item_id, decision)
            return RelayOutcome(decision.action, decision.reason, stopped=True, notified=notified)
        return RelayOutcome(decision.action, decision.reason)

    async def evaluate_and_apply(
        self, mission_id: str, item_id: str, ctx: RelayContext
    ) -> RelayOutcome:
        """Convenience: decide for ``ctx`` then apply. The one call a loop needs."""
        return await self.apply(mission_id, item_id, self.evaluate(ctx))

    async def _maybe_notify(self, mission_id: str, item_id: str, decision: RelayDecision) -> bool:
        if not decision.notify or self.notifier is None:
            return False
        channel = _CHANNEL_MAP.get(self.policy.relay_channel)
        if channel is None:  # relay_channel == "none"
            return False
        severity = "warning" if decision.action is RelayAction.PARK_AND_NOTIFY else "error"
        notification = Notification(
            title=f"[mission {mission_id}] {decision.action.value}",
            message=f"item {item_id}: {decision.reason}",
            severity=severity,
        )
        await self.notifier.notify(notification, channels=[channel])
        return True


__all__ = ["MissionRelay", "RelayOutcome"]
