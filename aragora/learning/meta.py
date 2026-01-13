"""
Meta-Learning module for self-tuning hyperparameters.

Implements the outer optimization loop of Nested Learning that
adjusts the learning system's own hyperparameters based on
performance metrics.

This enables the system to:
1. Tune tier promotion/demotion thresholds
2. Adjust surprise calculation weights
3. Modify decay half-lives
4. Optimize consolidation criteria
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List

from aragora.config import DB_MEMORY_PATH, DB_TIMEOUT_SECONDS
from aragora.storage.base_store import SQLiteStore
from aragora.utils.json_helpers import safe_json_loads

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for evaluating learning efficiency."""

    cycles_evaluated: int = 0
    pattern_retention_rate: float = 0.0  # % of patterns still useful
    forgetting_rate: float = 0.0  # Rate of useful pattern loss
    learning_velocity: float = 0.0  # Speed of new pattern acquisition
    consensus_rate: float = 0.0  # % of debates reaching consensus
    avg_cycles_to_consensus: float = 0.0
    prediction_accuracy: float = 0.0  # Agent calibration quality
    tier_efficiency: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycles_evaluated": self.cycles_evaluated,
            "pattern_retention_rate": self.pattern_retention_rate,
            "forgetting_rate": self.forgetting_rate,
            "learning_velocity": self.learning_velocity,
            "consensus_rate": self.consensus_rate,
            "avg_cycles_to_consensus": self.avg_cycles_to_consensus,
            "prediction_accuracy": self.prediction_accuracy,
            "tier_efficiency": self.tier_efficiency,
        }


@dataclass
class HyperparameterState:
    """Current state of tunable hyperparameters."""

    # Surprise calculation weights
    surprise_weight_success: float = 0.3
    surprise_weight_semantic: float = 0.3
    surprise_weight_temporal: float = 0.2
    surprise_weight_agent: float = 0.2

    # Tier thresholds
    fast_promotion_threshold: float = 0.7
    medium_promotion_threshold: float = 0.6
    slow_promotion_threshold: float = 0.5
    fast_demotion_threshold: float = 0.2
    medium_demotion_threshold: float = 0.3
    slow_demotion_threshold: float = 0.4

    # Decay parameters
    fast_half_life_hours: float = 1.0
    medium_half_life_hours: float = 24.0
    slow_half_life_hours: float = 168.0
    glacial_half_life_hours: float = 720.0

    # Consolidation
    consolidation_threshold: int = 100
    promotion_cooldown_hours: float = 24.0

    # Learning rates
    meta_learning_rate: float = 0.01  # Rate at which hyperparams change

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surprise_weight_success": self.surprise_weight_success,
            "surprise_weight_semantic": self.surprise_weight_semantic,
            "surprise_weight_temporal": self.surprise_weight_temporal,
            "surprise_weight_agent": self.surprise_weight_agent,
            "fast_promotion_threshold": self.fast_promotion_threshold,
            "medium_promotion_threshold": self.medium_promotion_threshold,
            "slow_promotion_threshold": self.slow_promotion_threshold,
            "fast_demotion_threshold": self.fast_demotion_threshold,
            "medium_demotion_threshold": self.medium_demotion_threshold,
            "slow_demotion_threshold": self.slow_demotion_threshold,
            "fast_half_life_hours": self.fast_half_life_hours,
            "medium_half_life_hours": self.medium_half_life_hours,
            "slow_half_life_hours": self.slow_half_life_hours,
            "glacial_half_life_hours": self.glacial_half_life_hours,
            "consolidation_threshold": self.consolidation_threshold,
            "promotion_cooldown_hours": self.promotion_cooldown_hours,
            "meta_learning_rate": self.meta_learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterState":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class MetaLearner(SQLiteStore):
    """
    Outer optimization loop that tunes the learning system itself.

    The MetaLearner observes the performance of the ContinuumMemory
    system and adjusts hyperparameters to improve learning efficiency.

    Key adjustments:
    1. Tier thresholds: If too many promotions, raise threshold
    2. Surprise weights: Balance based on prediction accuracy
    3. Decay half-lives: Adjust based on retention needs
    4. Consolidation: Speed up/slow down based on stability

    Inherits from SQLiteStore for standardized schema management.

    Usage:
        meta = MetaLearner()

        # After each nomic cycle, evaluate and adjust
        metrics = meta.evaluate_learning_efficiency(cms, cycle_results)
        adjustments = meta.adjust_hyperparameters(metrics)

        # Apply adjustments to ContinuumMemory
        cms.hyperparams.update(meta.get_current_hyperparams())
    """

    SCHEMA_NAME = "meta_learner"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Hyperparameter history table
        CREATE TABLE IF NOT EXISTS meta_hyperparams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hyperparams TEXT NOT NULL,
            metrics TEXT,
            adjustment_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Learning efficiency history
        CREATE TABLE IF NOT EXISTS meta_efficiency_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_number INTEGER,
            metrics TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """

    def __init__(self, db_path: str = DB_MEMORY_PATH):
        super().__init__(db_path, timeout=DB_TIMEOUT_SECONDS)
        self.state = self._load_state()
        self.metrics_history: List[LearningMetrics] = []

    def _load_state(self) -> HyperparameterState:
        """Load the most recent hyperparameter state.

        Returns default state if database is unavailable or corrupted.
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT hyperparams FROM meta_hyperparams
                    ORDER BY created_at DESC LIMIT 1
                """
                )
                row = cursor.fetchone()

            if row:
                data: dict[str, Any] = safe_json_loads(row[0], {})
                if data:
                    return HyperparameterState.from_dict(data)
        except sqlite3.Error as e:
            logger.warning(f"Failed to load hyperparameter state: {e}")
            # Fall through to return defaults

        return HyperparameterState()  # Default state

    def _save_state(self, reason: str = "", metrics: LearningMetrics | None = None):
        """Save current hyperparameter state.

        Logs warning if save fails but doesn't raise - in-memory state is preserved.
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO meta_hyperparams (hyperparams, metrics, adjustment_reason)
                    VALUES (?, ?, ?)
                    """,
                    (
                        json.dumps(self.state.to_dict()),
                        json.dumps(metrics.to_dict()) if metrics else None,
                        reason,
                    ),
                )

                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to save hyperparameter state: {e}")
            # Continue with in-memory state - next save may succeed

    def get_current_hyperparams(self) -> Dict[str, Any]:
        """Get current hyperparameters for ContinuumMemory."""
        return {
            "surprise_weight_success": self.state.surprise_weight_success,
            "surprise_weight_semantic": self.state.surprise_weight_semantic,
            "surprise_weight_temporal": self.state.surprise_weight_temporal,
            "surprise_weight_agent": self.state.surprise_weight_agent,
            "consolidation_threshold": self.state.consolidation_threshold,
            "promotion_cooldown_hours": self.state.promotion_cooldown_hours,
        }

    def evaluate_learning_efficiency(
        self,
        cms,  # ContinuumMemory instance
        cycle_results: Dict[str, Any],
    ) -> LearningMetrics:
        """
        Evaluate how well the learning system is performing.

        Computes metrics from:
        1. Pattern usage and success rates
        2. Tier distribution and transitions
        3. Debate outcomes
        4. Agent prediction accuracy

        Args:
            cms: ContinuumMemory instance to evaluate
            cycle_results: Results from the most recent nomic cycle

        Returns:
            LearningMetrics with computed efficiency scores.
            Returns partial metrics if DB queries fail.
        """
        metrics = LearningMetrics()

        # Get CMS stats
        try:
            cms_stats = cms.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get CMS stats: {e}")
            cms_stats = {}
        total_memories = cms_stats.get("total_memories", 0)

        if total_memories == 0:
            return metrics

        # Query pattern statistics from database
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                # Pattern retention: % of patterns with success_rate > 0.5
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM continuum_memory
                    WHERE (success_count * 1.0 / NULLIF(success_count + failure_count, 0)) > 0.5
                """
                )
                row = cursor.fetchone()
                useful_count = (row[0] if row else 0) or 0
                metrics.pattern_retention_rate = (
                    useful_count / total_memories if total_memories > 0 else 0
                )

                # Forgetting rate: % of patterns that became less useful over time
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM continuum_memory
                    WHERE failure_count > success_count
                      AND update_count > 5
                """
                )
                row = cursor.fetchone()
                forgotten_count = (row[0] if row else 0) or 0
                metrics.forgetting_rate = (
                    forgotten_count / total_memories if total_memories > 0 else 0
                )

                # Tier efficiency: success rate per tier
                for tier in ["fast", "medium", "slow", "glacial"]:
                    cursor.execute(
                        """
                        SELECT AVG(success_count * 1.0 / NULLIF(success_count + failure_count, 0))
                        FROM continuum_memory
                        WHERE tier = ? AND (success_count + failure_count) > 0
                        """,
                        (tier,),
                    )
                    row = cursor.fetchone()
                    result = row[0] if row else None
                    metrics.tier_efficiency[tier] = result or 0.5

                # Learning velocity: new patterns per cycle
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM continuum_memory
                    WHERE julianday('now') - julianday(created_at) < 1
                """
                )
                row = cursor.fetchone()
                metrics.learning_velocity = (row[0] if row else 0) or 0
        except sqlite3.Error as e:
            logger.warning(f"Failed to query learning metrics from DB: {e}")
            # Continue with default values in metrics

        # Extract from cycle results
        metrics.cycles_evaluated = cycle_results.get("cycle", 0)
        metrics.consensus_rate = cycle_results.get("consensus_rate", 0.5)
        metrics.prediction_accuracy = cycle_results.get("avg_calibration", 0.5)

        # Store in history
        self.metrics_history.append(metrics)

        # Log to database
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO meta_efficiency_log (cycle_number, metrics)
                    VALUES (?, ?)
                    """,
                    (metrics.cycles_evaluated, json.dumps(metrics.to_dict())),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to log efficiency metrics to DB: {e}")
            # Continue - in-memory history is preserved

        return metrics

    def adjust_hyperparameters(self, metrics: LearningMetrics) -> Dict[str, Any]:
        """
        Self-modify learning parameters based on performance.

        Adjustment rules:
        1. Low retention → increase half-lives (slower decay)
        2. High forgetting → lower promotion thresholds (faster learning)
        3. Tier imbalance → adjust tier-specific thresholds
        4. Poor calibration → adjust agent weight in surprise

        Args:
            metrics: Current learning efficiency metrics

        Returns:
            Dict of adjusted parameters and reasons
        """
        adjustments = {}
        lr = self.state.meta_learning_rate

        # Rule 1: Retention adjustment
        if metrics.pattern_retention_rate < 0.6:
            # Too much forgetting - slow down decay
            self.state.slow_half_life_hours *= 1 + lr
            self.state.glacial_half_life_hours *= 1 + lr
            adjustments["half_lives"] = "increased (low retention)"
        elif metrics.pattern_retention_rate > 0.9:
            # Very high retention - can afford faster decay
            self.state.slow_half_life_hours *= 1 - lr * 0.5
            adjustments["half_lives"] = "decreased (high retention)"

        # Rule 2: Forgetting rate adjustment
        if metrics.forgetting_rate > 0.3:
            # Too many patterns failing - lower promotion threshold
            self.state.medium_promotion_threshold *= 1 - lr
            self.state.slow_promotion_threshold *= 1 - lr
            adjustments["promotion_thresholds"] = "lowered (high forgetting)"
        elif metrics.forgetting_rate < 0.1:
            # Very stable - can be more selective
            self.state.medium_promotion_threshold *= 1 + lr * 0.5
            adjustments["promotion_thresholds"] = "raised (low forgetting)"

        # Rule 3: Tier efficiency balancing
        tier_eff = metrics.tier_efficiency
        if tier_eff.get("fast", 0.5) < tier_eff.get("slow", 0.5) - 0.1:
            # Fast tier underperforming - raise bar for promotion
            self.state.fast_promotion_threshold *= 1 + lr
            adjustments["fast_threshold"] = "raised (fast tier underperforming)"

        # Rule 4: Calibration adjustment
        if metrics.prediction_accuracy < 0.4:
            # Poor agent calibration - reduce agent weight
            self.state.surprise_weight_agent *= 1 - lr
            # Redistribute to success weight
            self.state.surprise_weight_success += lr * 0.1
            adjustments["surprise_weights"] = "reduced agent weight (poor calibration)"
        elif metrics.prediction_accuracy > 0.7:
            # Good calibration - increase agent weight
            self.state.surprise_weight_agent *= 1 + lr * 0.5
            adjustments["surprise_weights"] = "increased agent weight (good calibration)"

        # Normalize surprise weights to sum to 1.0
        total_weight = (
            self.state.surprise_weight_success
            + self.state.surprise_weight_semantic
            + self.state.surprise_weight_temporal
            + self.state.surprise_weight_agent
        )
        if total_weight > 0:
            self.state.surprise_weight_success /= total_weight
            self.state.surprise_weight_semantic /= total_weight
            self.state.surprise_weight_temporal /= total_weight
            self.state.surprise_weight_agent /= total_weight

        # Clamp values to reasonable ranges
        self._clamp_hyperparameters()

        # Save updated state
        if adjustments:
            reason = "; ".join(f"{k}: {v}" for k, v in adjustments.items())
            self._save_state(reason=reason, metrics=metrics)

        return adjustments

    def _clamp_hyperparameters(self):
        """Ensure hyperparameters stay in valid ranges."""
        # Thresholds: 0.1 to 0.9
        self.state.fast_promotion_threshold = max(
            0.1, min(0.9, self.state.fast_promotion_threshold)
        )
        self.state.medium_promotion_threshold = max(
            0.1, min(0.9, self.state.medium_promotion_threshold)
        )
        self.state.slow_promotion_threshold = max(
            0.1, min(0.9, self.state.slow_promotion_threshold)
        )
        self.state.fast_demotion_threshold = max(0.1, min(0.9, self.state.fast_demotion_threshold))
        self.state.medium_demotion_threshold = max(
            0.1, min(0.9, self.state.medium_demotion_threshold)
        )
        self.state.slow_demotion_threshold = max(0.1, min(0.9, self.state.slow_demotion_threshold))

        # Half-lives: minimum 0.5 hours, maximum 2000 hours
        self.state.fast_half_life_hours = max(0.5, min(24, self.state.fast_half_life_hours))
        self.state.medium_half_life_hours = max(6, min(168, self.state.medium_half_life_hours))
        self.state.slow_half_life_hours = max(24, min(720, self.state.slow_half_life_hours))
        self.state.glacial_half_life_hours = max(168, min(2000, self.state.glacial_half_life_hours))

        # Weights: 0.05 to 0.6
        self.state.surprise_weight_success = max(0.05, min(0.6, self.state.surprise_weight_success))
        self.state.surprise_weight_semantic = max(
            0.05, min(0.6, self.state.surprise_weight_semantic)
        )
        self.state.surprise_weight_temporal = max(
            0.05, min(0.6, self.state.surprise_weight_temporal)
        )
        self.state.surprise_weight_agent = max(0.05, min(0.6, self.state.surprise_weight_agent))

        # Meta learning rate: 0.001 to 0.1
        self.state.meta_learning_rate = max(0.001, min(0.1, self.state.meta_learning_rate))

    def get_adjustment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent hyperparameter adjustments.

        Returns empty list if database is unavailable.
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT hyperparams, metrics, adjustment_reason, created_at
                    FROM meta_hyperparams
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                history = []
                for row in cursor.fetchall():
                    history.append(
                        {
                            "hyperparams": safe_json_loads(row[0], {}),
                            "metrics": safe_json_loads(row[1], None),
                            "reason": row[2],
                            "timestamp": row[3],
                        }
                    )

            return history
        except sqlite3.Error as e:
            logger.warning(f"Failed to get adjustment history: {e}")
            return []

    def reset_to_defaults(self) -> None:
        """Reset hyperparameters to default values."""
        self.state = HyperparameterState()
        self._save_state(reason="reset to defaults")

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of meta-learning performance."""
        if not self.metrics_history:
            return {"status": "no data"}

        recent = self.metrics_history[-10:]  # Last 10 evaluations

        return {
            "evaluations": len(self.metrics_history),
            "avg_retention": sum(m.pattern_retention_rate for m in recent) / len(recent),
            "avg_forgetting": sum(m.forgetting_rate for m in recent) / len(recent),
            "avg_learning_velocity": sum(m.learning_velocity for m in recent) / len(recent),
            "current_hyperparams": self.state.to_dict(),
            "trend": self._compute_trend(recent),
        }

    def _compute_trend(self, recent_metrics: List[LearningMetrics]) -> str:
        """Compute overall learning trend from recent metrics."""
        if len(recent_metrics) < 2:
            return "insufficient_data"

        # Compare first half vs second half
        mid = len(recent_metrics) // 2
        first_half = recent_metrics[:mid]
        second_half = recent_metrics[mid:]

        # Defensive check for empty halves (shouldn't happen with len >= 2 check above)
        if not first_half or not second_half:
            return "insufficient_data"

        first_retention = sum(m.pattern_retention_rate for m in first_half) / len(first_half)
        second_retention = sum(m.pattern_retention_rate for m in second_half) / len(second_half)

        if second_retention > first_retention + 0.05:
            return "improving"
        elif second_retention < first_retention - 0.05:
            return "declining"
        else:
            return "stable"
