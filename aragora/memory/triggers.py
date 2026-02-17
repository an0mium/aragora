"""
MemoryTriggerEngine -- Reactive rules fired on memory events.

Make.com-style triggers that fire workflows when memory events occur
(high surprise, stale knowledge, contradiction, consolidation, new pattern).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class MemoryTrigger:
    """A single reactive trigger rule."""

    name: str
    event: str  # "high_surprise", "stale_knowledge", "contradiction", "consolidation", "new_pattern", "query_result", "new_write"
    condition: Callable[[dict[str, Any]], bool] | None = None
    action: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    enabled: bool = True


@dataclass
class TriggerResult:
    """Result of firing a single trigger."""

    trigger_name: str
    success: bool
    error: str | None = None


class MemoryTriggerEngine:
    """Reactive rule engine fired on memory events.

    Manages a registry of triggers that match on event type, optionally
    filter by condition, and execute async action callbacks. Errors in
    one trigger never block others.
    """

    def __init__(self) -> None:
        self._triggers: dict[str, MemoryTrigger] = {}
        self._fire_log: list[TriggerResult] = []
        self._register_builtins()

    def register(self, trigger: MemoryTrigger) -> None:
        """Register a trigger by name (overwrites if exists)."""
        self._triggers[trigger.name] = trigger

    def unregister(self, name: str) -> bool:
        """Remove a trigger by name. Returns True if it existed."""
        return self._triggers.pop(name, None) is not None

    async def fire(self, event: str, context: dict[str, Any]) -> list[str]:
        """Fire all triggers matching *event*, return list of triggered names.

        Each matching trigger's condition is checked (if present). If the
        condition passes (or is None), the action is executed.

        Errors in one trigger do not block others -- they are logged and
        recorded in the fire log.
        """
        triggered: list[str] = []

        for trigger in self._triggers.values():
            if not trigger.enabled:
                continue
            if trigger.event != event:
                continue

            # Check condition
            if trigger.condition is not None:
                try:
                    if not trigger.condition(context):
                        continue
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    logger.warning(
                        "Trigger %s condition failed: %s", trigger.name, exc
                    )
                    self._fire_log.append(
                        TriggerResult(
                            trigger_name=trigger.name,
                            success=False,
                            error=f"condition error: {exc}",
                        )
                    )
                    continue

            # Execute action
            if trigger.action is not None:
                try:
                    await trigger.action(context)
                    self._fire_log.append(
                        TriggerResult(trigger_name=trigger.name, success=True)
                    )
                    triggered.append(trigger.name)
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                    logger.warning(
                        "Trigger %s action failed: %s", trigger.name, exc
                    )
                    self._fire_log.append(
                        TriggerResult(
                            trigger_name=trigger.name,
                            success=False,
                            error=f"action error: {exc}",
                        )
                    )
                    triggered.append(trigger.name)
            else:
                # No action -- trigger matched but has no handler
                self._fire_log.append(
                    TriggerResult(trigger_name=trigger.name, success=True)
                )
                triggered.append(trigger.name)

        return triggered

    def list_triggers(self) -> list[MemoryTrigger]:
        """Return all registered triggers."""
        return list(self._triggers.values())

    def get_trigger(self, name: str) -> MemoryTrigger | None:
        """Get a trigger by name."""
        return self._triggers.get(name)

    def enable(self, name: str) -> None:
        """Enable a trigger by name."""
        if name in self._triggers:
            self._triggers[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a trigger by name."""
        if name in self._triggers:
            self._triggers[name].enabled = False

    def get_fire_log(self) -> list[TriggerResult]:
        """Return a copy of the fire log."""
        return list(self._fire_log)

    def clear_fire_log(self) -> None:
        """Clear the fire log."""
        self._fire_log.clear()

    def _register_builtins(self) -> None:
        """Register the 5 built-in triggers."""
        self.register(
            MemoryTrigger(
                name="high_surprise_investigate",
                event="high_surprise",
                condition=lambda ctx: ctx.get("surprise", 0) > 0.7,
                action=_log_high_surprise,
            )
        )
        self.register(
            MemoryTrigger(
                name="stale_knowledge_revalidate",
                event="stale_knowledge",
                condition=lambda ctx: (
                    ctx.get("days_since_access", 0) > 7
                    and ctx.get("confidence", 1.0) < 0.5
                ),
                action=_mark_for_revalidation,
            )
        )
        self.register(
            MemoryTrigger(
                name="contradiction_detected",
                event="contradiction",
                condition=None,
                action=_create_debate_topic,
            )
        )
        self.register(
            MemoryTrigger(
                name="consolidation_merge",
                event="consolidation",
                condition=lambda ctx: (
                    ctx.get("item_count", 0) >= 3
                    and ctx.get("avg_surprise", 1.0) < 0.2
                ),
                action=_merge_summaries,
            )
        )
        self.register(
            MemoryTrigger(
                name="pattern_emergence",
                event="new_pattern",
                condition=lambda ctx: ctx.get("surprise_ema_trend") == "decreasing",
                action=_extract_pattern,
            )
        )


# -----------------------------------------------------------------------
# Built-in action functions
# -----------------------------------------------------------------------


async def _log_high_surprise(context: dict[str, Any]) -> None:
    logger.info(
        "High surprise detected: item_id=%s surprise=%.3f",
        context.get("item_id"),
        context.get("surprise", 0),
    )


async def _mark_for_revalidation(context: dict[str, Any]) -> None:
    logger.info(
        "Stale knowledge marked for revalidation: item_id=%s",
        context.get("item_id"),
    )


async def _create_debate_topic(context: dict[str, Any]) -> None:
    logger.info(
        "Contradiction detected, debate topic created: %s",
        context.get("description", ""),
    )


async def _merge_summaries(context: dict[str, Any]) -> None:
    logger.info(
        "Consolidation merge triggered for %d items",
        context.get("item_count", 0),
    )


async def _extract_pattern(context: dict[str, Any]) -> None:
    logger.info(
        "Pattern emergence detected: %s",
        context.get("pattern", ""),
    )


__all__ = ["MemoryTriggerEngine", "MemoryTrigger", "TriggerResult"]
