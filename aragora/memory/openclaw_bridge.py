"""OpenClaw <-> Continuum Memory bridge.

Stores OpenClaw validation events in ContinuumMemory at the medium tier,
letting the retention gate naturally manage validation history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationEvent:
    agent_id: str
    status: str  # "pass", "fail", "revoke"
    details: str = ""
    confidence: float = 0.8


class OpenClawContinuumBridge:
    def __init__(self, continuum_memory: Any):
        self._continuum = continuum_memory
        self._event_count = 0

    def record_validation(self, event: ValidationEvent) -> str | None:
        """Record validation event in Continuum at medium tier."""
        content = f"OpenClaw validation [{event.status}] agent={event.agent_id}: {event.details}"
        importance = {
            "pass": 0.4,
            "fail": 0.7,
            "revoke": 0.9,
        }.get(event.status, 0.5)

        try:
            if hasattr(self._continuum, "store_pattern"):
                entry_id = self._continuum.store_pattern(
                    content=content,
                    importance=importance,
                    metadata={
                        "agent_id": event.agent_id,
                        "status": event.status,
                        "source": "openclaw_bridge",
                        "tier_hint": "medium",
                    },
                )
                self._event_count += 1
                return entry_id
            elif hasattr(self._continuum, "add"):
                entry = self._continuum.add(
                    id=f"oclaw_{event.agent_id}_{self._event_count}",
                    content=content,
                    importance=importance,
                    metadata={"agent_id": event.agent_id, "source": "openclaw_bridge"},
                )
                self._event_count += 1
                return entry.id
        except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
            logger.warning("OpenClaw-Continuum bridge failed: %s", exc)
        return None

    def get_event_count(self) -> int:
        return self._event_count
