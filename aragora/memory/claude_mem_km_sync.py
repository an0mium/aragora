"""Claude-Mem <-> Knowledge Mound synchronization.

Periodically extracts high-confidence insights from Claude-Mem
conversation history and writes them as KM entries.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from aragora.memory.surprise import ContentSurpriseScorer

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    synced_count: int
    skipped_count: int
    errors: int


class ClaudeMemKMSync:
    def __init__(
        self,
        claude_mem_backend: Any,  # Has search(query, limit) method
        km_backend: Any,  # Has store_knowledge(content, source, source_id, confidence, metadata)
        surprise_scorer: ContentSurpriseScorer | None = None,
        min_surprise: float = 0.4,  # Only sync items above this surprise threshold
    ):
        self._claude_mem = claude_mem_backend
        self._km = km_backend
        self._scorer = surprise_scorer or ContentSurpriseScorer()
        self._min_surprise = min_surprise
        self._synced_ids: set[str] = set()

    async def sync(self, query: str = "important insights", limit: int = 50) -> SyncResult:
        """Extract high-surprise insights from Claude-Mem and write to KM."""
        synced = 0
        skipped = 0
        errors = 0

        # Query Claude-Mem
        try:
            if hasattr(self._claude_mem, "search"):
                search_fn = getattr(self._claude_mem, "search")
                if asyncio.iscoroutinefunction(search_fn):
                    results = await search_fn(query, limit=limit)
                else:
                    results = search_fn(query, limit=limit)
            else:
                return SyncResult(0, 0, 0)
        except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
            logger.warning("Claude-Mem query failed: %s", exc)
            return SyncResult(0, 0, 1)

        for item in results or []:
            item_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")
            content = (
                item.get("content", "") if isinstance(item, dict) else getattr(item, "content", "")
            )

            if not content or item_id in self._synced_ids:
                skipped += 1
                continue

            # Score surprise
            score = self._scorer.score(content, "claude_mem")
            if score.combined < self._min_surprise:
                skipped += 1
                continue

            # Write to KM
            try:
                source_id = item_id or f"cm_{hash(content) % 10**8}"
                if hasattr(self._km, "store_knowledge"):
                    store_fn = getattr(self._km, "store_knowledge")
                    if asyncio.iscoroutinefunction(store_fn):
                        await store_fn(
                            content=content,
                            source="claude_mem_sync",
                            source_id=source_id,
                            confidence=min(1.0, score.combined),
                            metadata={"synced_from": "claude_mem", "surprise": score.combined},
                        )
                    else:
                        store_fn(
                            content=content,
                            source="claude_mem_sync",
                            source_id=source_id,
                            confidence=min(1.0, score.combined),
                            metadata={"synced_from": "claude_mem", "surprise": score.combined},
                        )
                self._synced_ids.add(item_id)
                synced += 1
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
                logger.warning("KM write failed for item %s: %s", item_id, exc)
                errors += 1

        return SyncResult(synced, skipped, errors)

    def reset_sync_tracking(self) -> None:
        self._synced_ids.clear()
