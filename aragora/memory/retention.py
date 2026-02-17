"""
MIRAS-inspired retention gate coordinating decay across all memory systems.

Each system already has its own decay/TTL (Continuum has tier-based,
Consensus has none, KM has none, Supermemory has weight threshold).

The RetentionGate adds a UNIFIED layer:
- Scores each stored item by: recency × access_frequency × surprise_score
- Items below retention threshold get marked for demotion/archival
- Runs periodically as a background sweep

Reference: Titans + MIRAS - Helping AI have long-term memory (Google Research)
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RetentionAction(Enum):
    """What the gate recommends for an item."""

    KEEP = "keep"
    DEMOTE = "demote"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class RetentionDecision:
    """Result of evaluating a single item's retention."""

    item_id: str
    system: str
    action: RetentionAction
    score: float  # 0-1 composite retention score
    reason: str


@dataclass(frozen=True)
class MergeAction:
    """Recommendation to merge near-duplicate items across systems."""

    primary_id: str
    primary_system: str
    duplicate_id: str
    duplicate_system: str
    similarity: float
    recommendation: str


@dataclass
class RetentionReport:
    """Aggregated results of a retention sweep."""

    evaluated: int = 0
    keep: int = 0
    demote: int = 0
    archive: int = 0
    decisions: list[RetentionDecision] = field(default_factory=list)
    merge_suggestions: list[MergeAction] = field(default_factory=list)


@dataclass
class ItemMetadata:
    """Metadata used for retention scoring."""

    item_id: str
    system: str
    created_at: float  # unix timestamp
    last_accessed: float  # unix timestamp
    access_count: int = 1
    surprise_score: float = 0.5
    importance: float = 0.5


class RetentionGate:
    """Coordinated retention gate across all memory systems.

    Does NOT delete — returns recommendations for the coordinator to act on.
    """

    def __init__(
        self,
        demote_threshold: float = 0.3,
        archive_threshold: float = 0.15,
        half_life_days: float = 30.0,
    ):
        self._demote_threshold = demote_threshold
        self._archive_threshold = archive_threshold
        self._half_life_days = half_life_days

    def evaluate(self, item: ItemMetadata) -> RetentionDecision:
        """Evaluate whether an item should be kept, demoted, or archived.

        Composite score = recency × access_frequency × surprise_score
        """
        now = time.time()

        # Recency: exponential decay with configurable half-life
        age_days = max(0.0, (now - item.last_accessed) / 86400.0)
        recency = 0.5 ** (age_days / self._half_life_days)

        # Access frequency: log-scaled, normalised to 0-1
        frequency = min(1.0, math.log1p(item.access_count) / math.log1p(100))

        # Composite
        score = round(
            0.4 * recency + 0.3 * frequency + 0.3 * item.surprise_score,
            4,
        )

        if score >= self._demote_threshold:
            action = RetentionAction.KEEP
            reason = f"Retention score {score:.3f} above demote threshold"
        elif score >= self._archive_threshold:
            action = RetentionAction.DEMOTE
            reason = (
                f"Retention score {score:.3f} below demote threshold "
                f"({self._demote_threshold}), recommend demotion"
            )
        else:
            action = RetentionAction.ARCHIVE
            reason = (
                f"Retention score {score:.3f} below archive threshold "
                f"({self._archive_threshold}), recommend archival"
            )

        return RetentionDecision(
            item_id=item.item_id,
            system=item.system,
            action=action,
            score=score,
            reason=reason,
        )

    def sweep(self, items: list[ItemMetadata]) -> RetentionReport:
        """Evaluate a batch of items and produce a retention report."""
        report = RetentionReport()

        for item in items:
            decision = self.evaluate(item)
            report.decisions.append(decision)
            report.evaluated += 1

            if decision.action == RetentionAction.KEEP:
                report.keep += 1
            elif decision.action == RetentionAction.DEMOTE:
                report.demote += 1
            else:
                report.archive += 1

        logger.info(
            "Retention sweep: evaluated=%d keep=%d demote=%d archive=%d",
            report.evaluated,
            report.keep,
            report.demote,
            report.archive,
        )
        return report

    def consolidate_duplicates(
        self,
        items_by_system: dict[str, list[dict[str, Any]]],
    ) -> list[MergeAction]:
        """Find near-duplicate items across systems and recommend merges.

        Uses lightweight keyword overlap to identify candidates without
        requiring embedding infrastructure.

        Args:
            items_by_system: Mapping of system name → list of item dicts
                each with at least "id" and "content" keys.
        """
        actions: list[MergeAction] = []
        all_items: list[tuple[str, str, set[str]]] = []

        for system, items in items_by_system.items():
            for item in items:
                kw = {
                    w
                    for w in re.findall(r"[a-z]{3,}", item.get("content", "").lower())
                    if len(w) >= 3
                }
                if kw:
                    all_items.append((system, item["id"], kw))

        # O(n²) but items list is expected to be small (batch-limited)
        for i, (sys_a, id_a, kw_a) in enumerate(all_items):
            for sys_b, id_b, kw_b in all_items[i + 1 :]:
                if sys_a == sys_b:
                    continue
                overlap = len(kw_a & kw_b)
                union = len(kw_a | kw_b)
                if union == 0:
                    continue
                similarity = overlap / union
                if similarity >= 0.7:
                    actions.append(
                        MergeAction(
                            primary_id=id_a,
                            primary_system=sys_a,
                            duplicate_id=id_b,
                            duplicate_system=sys_b,
                            similarity=round(similarity, 3),
                            recommendation=f"High similarity ({similarity:.0%}), consider merging",
                        )
                    )

        return actions


__all__ = [
    "RetentionGate",
    "RetentionDecision",
    "RetentionAction",
    "RetentionReport",
    "MergeAction",
    "ItemMetadata",
]
