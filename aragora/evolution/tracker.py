"""
Evolution performance tracker.

Tracks agent performance across generations to measure
the effectiveness of prompt evolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from aragora.config import DB_TIMEOUT_SECONDS, resolve_db_path
from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class OutcomeRecord:
    """A single debate outcome record."""

    agent: str
    won: bool
    debate_id: Optional[str] = None
    generation: int = 0
    recorded_at: str = ""

    def __post_init__(self):
        if not self.recorded_at:
            self.recorded_at = datetime.utcnow().isoformat()


class EvolutionTracker(SQLiteStore):
    """
    Tracks evolution performance metrics.

    Records debate outcomes per agent and generation, enabling
    analysis of whether evolved prompts actually improve performance.

    Inherits from SQLiteStore for standardized schema management.
    """

    SCHEMA_NAME = "evolution_tracker"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            won INTEGER NOT NULL,
            debate_id TEXT,
            generation INTEGER DEFAULT 0,
            recorded_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_outcomes_agent
        ON outcomes(agent);

        CREATE INDEX IF NOT EXISTS idx_outcomes_generation
        ON outcomes(generation);
    """

    def __init__(self, db_path: str = "aragora_evolution_tracker.db"):
        """
        Initialize the evolution tracker.

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__(resolve_db_path(db_path), timeout=DB_TIMEOUT_SECONDS)

    def record_outcome(
        self,
        agent: str,
        won: bool,
        debate_id: Optional[str] = None,
        generation: int = 0,
    ):
        """
        Record a debate outcome.

        Args:
            agent: Agent name
            won: Whether the agent won
            debate_id: Optional debate ID
            generation: Evolution generation (default 0)
        """
        record = OutcomeRecord(
            agent=agent,
            won=won,
            debate_id=debate_id,
            generation=generation,
        )

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO outcomes (agent, won, debate_id, generation, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.agent,
                    1 if record.won else 0,
                    record.debate_id,
                    record.generation,
                    record.recorded_at,
                ),
            )

        logger.debug(
            f"Recorded outcome for {agent}: {'win' if won else 'loss'} " f"(gen={generation})"
        )

    def get_agent_stats(self, agent: str) -> dict:
        """
        Get statistics for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Dict with wins, losses, total, win_rate
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    SUM(won) as wins,
                    COUNT(*) - SUM(won) as losses,
                    COUNT(*) as total
                FROM outcomes
                WHERE agent = ?
                """,
                (agent,),
            )
            row = cursor.fetchone()

            wins = row[0] or 0
            losses = row[1] or 0
            total = row[2] or 0
            win_rate = wins / total if total > 0 else 0.0

        return {
            "agent": agent,
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_rate": win_rate,
        }

    def get_generation_metrics(self, generation: int) -> dict:
        """
        Get metrics for a specific generation.

        Args:
            generation: Generation number

        Returns:
            Dict with total_debates, wins, losses, win_rate, agents
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    SUM(won) as wins,
                    COUNT(*) - SUM(won) as losses,
                    COUNT(*) as total,
                    COUNT(DISTINCT agent) as unique_agents
                FROM outcomes
                WHERE generation = ?
                """,
                (generation,),
            )
            row = cursor.fetchone()

            wins = row[0] or 0
            losses = row[1] or 0
            total = row[2] or 0
            unique_agents = row[3] or 0
            win_rate = wins / total if total > 0 else 0.0

        return {
            "generation": generation,
            "total_debates": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "unique_agents": unique_agents,
        }

    def get_performance_delta(
        self,
        agent: str,
        gen1: int,
        gen2: int,
    ) -> dict:
        """
        Calculate performance change between generations.

        Args:
            agent: Agent name
            gen1: Earlier generation
            gen2: Later generation

        Returns:
            Dict with win_rate_delta, gen1_stats, gen2_stats
        """
        gen1_stats = self._get_agent_generation_stats(agent, gen1)
        gen2_stats = self._get_agent_generation_stats(agent, gen2)

        win_rate_delta = gen2_stats["win_rate"] - gen1_stats["win_rate"]

        return {
            "agent": agent,
            "gen1": gen1,
            "gen2": gen2,
            "gen1_win_rate": gen1_stats["win_rate"],
            "gen2_win_rate": gen2_stats["win_rate"],
            "win_rate_delta": win_rate_delta,
            "improved": win_rate_delta > 0,
        }

    def _get_agent_generation_stats(self, agent: str, generation: int) -> dict:
        """Get stats for an agent in a specific generation."""
        with self.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    SUM(won) as wins,
                    COUNT(*) as total
                FROM outcomes
                WHERE agent = ? AND generation = ?
                """,
                (agent, generation),
            )
            row = cursor.fetchone()

            wins = row[0] or 0
            total = row[1] or 0
            win_rate = wins / total if total > 0 else 0.0

        return {
            "wins": wins,
            "total": total,
            "win_rate": win_rate,
        }

    def get_all_agents(self) -> list[str]:
        """Get list of all agents with recorded outcomes."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT agent FROM outcomes ORDER BY agent")
            result = [row[0] for row in cursor.fetchall()]
        return result

    def get_generation_trend(self, agent: str, max_generations: int = 10) -> list[dict]:
        """
        Get win rate trend across generations for an agent.

        Args:
            agent: Agent name
            max_generations: Maximum generations to include

        Returns:
            List of dicts with generation and win_rate
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT generation
                FROM outcomes
                WHERE agent = ?
                GROUP BY generation
                ORDER BY generation DESC
                LIMIT ?
                """,
                (agent, max_generations),
            )

            generations = [row[0] for row in cursor.fetchall()]

        return [
            {
                "generation": gen,
                **self._get_agent_generation_stats(agent, gen),
            }
            for gen in sorted(generations)
        ]
