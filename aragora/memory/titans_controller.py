"""
TitansMemoryController -- Active sweep controller for Titans-inspired memory management.

Wires RetentionGate + MemoryCoordinator + MemoryFabric into a continuous
sweep loop that executes retain/forget/consolidate/demote decisions.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aragora.memory.retention_gate import RetentionGate, RetentionGateConfig
from aragora.memory.surprise import ContentSurpriseScorer, update_surprise_ema

logger = logging.getLogger(__name__)


@dataclass
class SurpriseState:
    """Persistent surprise state for a single memory item."""

    item_id: str
    source: str
    surprise_ema: float
    last_access: float  # unix timestamp
    access_count: int
    updated_at: float


@dataclass
class SweepResult:
    """Result of a single sweep pass."""

    items_processed: int
    actions: dict[str, int]  # action -> count
    errors: int
    duration_seconds: float


class TitansMemoryController:
    """Active sweep controller for Titans-inspired memory management.

    Maintains per-item surprise state in SQLite and runs periodic sweeps
    through RetentionGate to decide retain/forget/consolidate/demote actions.

    Hooks into MemoryFabric via on_query() and on_write() callbacks.
    """

    def __init__(
        self,
        retention_gate: RetentionGate | None = None,
        surprise_scorer: ContentSurpriseScorer | None = None,
        db_path: str | Path | None = None,
        trigger_engine: Any | None = None,
    ) -> None:
        self._retention_gate = retention_gate or RetentionGate(RetentionGateConfig())
        self._surprise_scorer = surprise_scorer or ContentSurpriseScorer()
        self._trigger_engine = trigger_engine
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn = sqlite3.connect(self._db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the surprise_state table if it does not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS surprise_state (
                item_id TEXT NOT NULL,
                source TEXT NOT NULL,
                surprise_ema REAL NOT NULL DEFAULT 0.5,
                last_access REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (item_id, source)
            )
            """
        )
        self._conn.commit()

    def get_state(self, item_id: str, source: str) -> SurpriseState | None:
        """Read surprise state for a single item from SQLite."""
        cursor = self._conn.execute(
            "SELECT item_id, source, surprise_ema, last_access, access_count, updated_at "
            "FROM surprise_state WHERE item_id = ? AND source = ?",
            (item_id, source),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return SurpriseState(
            item_id=row[0],
            source=row[1],
            surprise_ema=row[2],
            last_access=row[3],
            access_count=row[4],
            updated_at=row[5],
        )

    def _upsert_state(self, state: SurpriseState) -> None:
        """Insert or replace surprise state in SQLite."""
        self._conn.execute(
            "INSERT OR REPLACE INTO surprise_state "
            "(item_id, source, surprise_ema, last_access, access_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                state.item_id,
                state.source,
                state.surprise_ema,
                state.last_access,
                state.access_count,
                state.updated_at,
            ),
        )
        self._conn.commit()

    def _list_states(self, batch_size: int = 100) -> list[SurpriseState]:
        """List states ordered by last_access ASC (oldest first)."""
        cursor = self._conn.execute(
            "SELECT item_id, source, surprise_ema, last_access, access_count, updated_at "
            "FROM surprise_state ORDER BY last_access ASC LIMIT ?",
            (batch_size,),
        )
        return [
            SurpriseState(
                item_id=row[0],
                source=row[1],
                surprise_ema=row[2],
                last_access=row[3],
                access_count=row[4],
                updated_at=row[5],
            )
            for row in cursor.fetchall()
        ]

    async def on_query(self, query: str, results: list[Any]) -> None:
        """Called after every MemoryFabric query.

        Updates surprise EMA and bumps access_count for each result.
        Fires ``query_result`` trigger if trigger_engine is set.
        """
        now = time.time()
        for result in results:
            item_id = getattr(result, "item_id", None) or ""
            source = getattr(result, "source_system", None) or ""
            if not item_id or not source:
                continue

            existing = self.get_state(item_id, source)
            if existing is not None:
                new_ema = update_surprise_ema(existing.surprise_ema, existing.surprise_ema * 0.9)
                self._upsert_state(
                    SurpriseState(
                        item_id=item_id,
                        source=source,
                        surprise_ema=new_ema,
                        last_access=now,
                        access_count=existing.access_count + 1,
                        updated_at=now,
                    )
                )
            else:
                self._upsert_state(
                    SurpriseState(
                        item_id=item_id,
                        source=source,
                        surprise_ema=0.5,
                        last_access=now,
                        access_count=1,
                        updated_at=now,
                    )
                )

            if self._trigger_engine:
                try:
                    await self._trigger_engine.fire(
                        "query_result",
                        {"item_id": item_id, "source": source, "query": query},
                    )
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
                    logger.warning("Trigger fire failed for query_result: %s", exc)

    async def on_write(self, item_id: str, source: str, content: str) -> None:
        """Called after every MemoryCoordinator write.

        Scores surprise via ContentSurpriseScorer.score() and stores
        the initial state in SQLite. Fires ``new_write`` or ``high_surprise``
        trigger depending on score.
        """
        now = time.time()
        score = self._surprise_scorer.score(content, source)

        self._upsert_state(
            SurpriseState(
                item_id=item_id,
                source=source,
                surprise_ema=score.combined,
                last_access=now,
                access_count=0,
                updated_at=now,
            )
        )

        if self._trigger_engine:
            event = "high_surprise" if score.combined >= 0.7 else "new_write"
            try:
                await self._trigger_engine.fire(
                    event,
                    {
                        "item_id": item_id,
                        "source": source,
                        "surprise": score.combined,
                        "novelty": score.novelty,
                        "content_preview": content[:200],
                    },
                )
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
                logger.warning("Trigger fire failed for %s: %s", event, exc)

    async def sweep(self, batch_size: int = 100) -> SweepResult:
        """Run a single sweep pass over items ordered by oldest access.

        Calls RetentionGate.evaluate() for each item and tallies actions.
        """
        start = time.time()
        states = self._list_states(batch_size)
        actions: dict[str, int] = {}
        errors = 0

        for state in states:
            try:
                decision = self._retention_gate.evaluate(
                    item_id=state.item_id,
                    source=state.source,
                    content="",
                    outcome_surprise=state.surprise_ema,
                    current_confidence=max(0.0, 1.0 - state.surprise_ema),
                    access_count=state.access_count,
                )

                action = decision.action
                actions[action] = actions.get(action, 0) + 1

                now = time.time()

                if action == "forget":
                    logger.info(
                        "Sweep: forget item_id=%s source=%s reason=%s",
                        state.item_id,
                        state.source,
                        decision.reason,
                    )
                    self._conn.execute(
                        "DELETE FROM surprise_state WHERE item_id = ? AND source = ?",
                        (state.item_id, state.source),
                    )
                    self._conn.commit()
                elif action == "demote":
                    self._upsert_state(
                        SurpriseState(
                            item_id=state.item_id,
                            source=state.source,
                            surprise_ema=state.surprise_ema * 0.8,
                            last_access=state.last_access,
                            access_count=state.access_count,
                            updated_at=now,
                        )
                    )
                elif action == "consolidate":
                    self._upsert_state(
                        SurpriseState(
                            item_id=state.item_id,
                            source=state.source,
                            surprise_ema=min(1.0, state.surprise_ema * 1.1),
                            last_access=state.last_access,
                            access_count=state.access_count,
                            updated_at=now,
                        )
                    )
                elif action == "retain":
                    if decision.decay_rate_override is not None:
                        self._upsert_state(
                            SurpriseState(
                                item_id=state.item_id,
                                source=state.source,
                                surprise_ema=state.surprise_ema,
                                last_access=state.last_access,
                                access_count=state.access_count,
                                updated_at=now,
                            )
                        )

                if self._trigger_engine and action in ("demote", "consolidate"):
                    try:
                        await self._trigger_engine.fire(
                            action if action == "consolidation" else action,
                            {
                                "item_id": state.item_id,
                                "source": state.source,
                                "action": action,
                                "surprise_ema": state.surprise_ema,
                            },
                        )
                    except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
                        logger.warning("Trigger fire failed during sweep: %s", exc)

            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as exc:
                logger.warning("Sweep error for item %s: %s", state.item_id, exc)
                errors += 1

        duration = time.time() - start
        return SweepResult(
            items_processed=len(states),
            actions=actions,
            errors=errors,
            duration_seconds=round(duration, 4),
        )

    async def run_sweep_loop(
        self,
        interval_seconds: float = 300.0,
        max_sweeps: int = 0,
    ) -> None:
        """Run sweep() in a background loop.

        Args:
            interval_seconds: Seconds between sweeps.
            max_sweeps: Maximum number of sweeps (0 = infinite).
        """
        sweeps_done = 0
        while True:
            try:
                result = await self.sweep()
                logger.info(
                    "Sweep completed: processed=%d actions=%s errors=%d duration=%.3fs",
                    result.items_processed,
                    result.actions,
                    result.errors,
                    result.duration_seconds,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Sweep loop iteration failed: %s", exc)

            sweeps_done += 1
            if max_sweeps > 0 and sweeps_done >= max_sweeps:
                break

            await asyncio.sleep(interval_seconds)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics from the surprise state table."""
        cursor = self._conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(surprise_ema), 0.0), "
            "COALESCE(MIN(surprise_ema), 0.0), COALESCE(MAX(surprise_ema), 0.0) "
            "FROM surprise_state"
        )
        row = cursor.fetchone()
        item_count = row[0] if row else 0
        avg_surprise = row[1] if row else 0.0
        min_surprise = row[2] if row else 0.0
        max_surprise = row[3] if row else 0.0

        return {
            "item_count": item_count,
            "avg_surprise": round(avg_surprise, 4),
            "min_surprise": round(min_surprise, 4),
            "max_surprise": round(max_surprise, 4),
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()


__all__ = ["TitansMemoryController", "SurpriseState", "SweepResult"]
