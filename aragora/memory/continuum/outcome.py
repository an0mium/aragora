"""
Outcome and surprise scoring mixin for ContinuumMemory.

Provides update_outcome, learning rate calculation, and related
surprise-based learning operations.
"""
# mypy: disable-error-code="misc"
# Mixin classes use self: "ContinuumMemory" type hints which mypy doesn't understand

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aragora.memory.tier_manager import MemoryTier, TierConfig
from aragora.resilience.retry import PROVIDER_RETRY_POLICIES, with_retry

from .types import TIER_CONFIGS

if TYPE_CHECKING:
    from .core import ContinuumMemory

logger = logging.getLogger(__name__)

# Retry configuration for memory operations
_MEMORY_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["memory"]


class OutcomeMixin:
    """Mixin providing outcome and surprise scoring operations for ContinuumMemory."""

    def update_outcome(
        self: "ContinuumMemory",
        id: str,
        success: bool,
        agent_prediction_error: float | None = None,
    ) -> float:
        """
        Update memory after observing outcome.

        This implements surprise-based learning: the surprise score is
        updated based on how unexpected the outcome was.

        Uses BEGIN IMMEDIATE to prevent race conditions by acquiring
        a write lock before reading the current state.

        Args:
            id: Memory ID
            success: Whether the pattern led to success
            agent_prediction_error: Optional agent's prediction error

        Returns:
            Updated surprise score
        """
        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()

            # Use BEGIN IMMEDIATE to acquire write lock before reading.
            # This prevents race conditions in read-modify-write operations.
            # Note: get_wal_connection sets busy_timeout (default 30s) to prevent
            # indefinite blocking - will raise sqlite3.OperationalError if timeout
            # exceeded waiting for the lock.
            cursor.execute("BEGIN IMMEDIATE")

            try:
                # Get current state (now protected by write lock)
                cursor.execute(
                    """
                    SELECT success_count, failure_count, surprise_score, tier
                    FROM continuum_memory WHERE id = ?
                    """,
                    (id,),
                )
                row: tuple[Any, ...] | None = cursor.fetchone()
                if not row:
                    cursor.execute("ROLLBACK")
                    return 0.0

                success_count: int
                failure_count: int
                old_surprise_raw: Any
                tier: str
                success_count, failure_count, old_surprise_raw, tier = row
                old_surprise: float = float(old_surprise_raw) if old_surprise_raw else 0.0
                total: int = success_count + failure_count

                # Calculate expected success rate (base rate)
                expected_rate: float = success_count / total if total > 0 else 0.5

                # Actual outcome
                actual: float = 1.0 if success else 0.0

                # Success rate surprise component
                success_surprise: float = abs(actual - expected_rate)

                # Combine surprise signals
                new_surprise: float = self.hyperparams[
                    "surprise_weight_success"
                ] * success_surprise + self.hyperparams["surprise_weight_agent"] * (
                    agent_prediction_error or 0.0
                )

                # Exponential moving average for surprise
                alpha: float = 0.3
                updated_surprise: float = old_surprise * (1 - alpha) + new_surprise * alpha

                # Update consolidation score
                update_count: int = total + 1
                consolidation: float = min(
                    1.0,
                    math.log(1 + update_count)
                    / math.log(self.hyperparams["consolidation_threshold"]),
                )

                # Update database
                if success:
                    cursor.execute(
                        """
                        UPDATE continuum_memory
                        SET success_count = success_count + 1,
                            update_count = update_count + 1,
                            surprise_score = ?,
                            consolidation_score = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (updated_surprise, consolidation, datetime.now().isoformat(), id),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE continuum_memory
                        SET failure_count = failure_count + 1,
                            update_count = update_count + 1,
                            surprise_score = ?,
                            consolidation_score = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (updated_surprise, consolidation, datetime.now().isoformat(), id),
                    )

                cursor.execute("COMMIT")
            except sqlite3.Error as e:
                logger.error(f"Database error updating surprise score: {e}", exc_info=True)
                cursor.execute("ROLLBACK")
                raise
            except (ValueError, TypeError, ArithmeticError, RuntimeError) as e:
                # Rollback on non-database exceptions, then re-raise unchanged
                logger.warning(
                    f"Non-database exception during surprise update, rolling back: {type(e).__name__}: {e}"
                )
                cursor.execute("ROLLBACK")
                raise

        return updated_surprise

    @with_retry(_MEMORY_RETRY_CONFIG)
    async def update_outcome_async(
        self: "ContinuumMemory",
        id: str,
        success: bool,
        agent_prediction_error: float | None = None,
    ) -> float:
        """Async wrapper for update_outcome() - offloads blocking I/O to executor."""
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.update_outcome(id, success, agent_prediction_error),
        )

    def get_learning_rate(self: "ContinuumMemory", tier: MemoryTier, update_count: int) -> float:
        """
        Get tier-specific learning rate with decay.

        HOPE-inspired: fast tiers have high initial LR with rapid decay,
        slow tiers have low initial LR with gradual decay.
        """
        config: TierConfig = TIER_CONFIGS[tier]
        return config.base_learning_rate * (config.decay_rate**update_count)


__all__ = ["OutcomeMixin"]
