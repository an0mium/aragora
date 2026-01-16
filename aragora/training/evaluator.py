"""
Evaluator for A/B testing fine-tuned models.

Compares fine-tuned Tinker models against baseline models using
debate outcomes, win rates, and calibration metrics.
"""

__all__ = [
    "ABTestResult",
    "EvaluationMetrics",
    "TinkerEvaluator",
]

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.core import Agent, DebateResult
from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Results of an A/B test between two agents."""

    test_id: str
    fine_tuned_agent: str
    baseline_agent: str
    num_trials: int
    fine_tuned_wins: int
    baseline_wins: int
    draws: int
    fine_tuned_win_rate: float
    avg_fine_tuned_score: float
    avg_baseline_score: float
    avg_confidence: float
    consensus_rate: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    trials: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        """Check if results are statistically significant (p < 0.05)."""
        # Simple binomial test approximation
        if self.num_trials < 10:
            return False
        # Wilson score interval
        n = self.num_trials
        p = self.fine_tuned_win_rate
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        spread = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5 / denominator
        lower = center - spread
        upper = center + spread
        # Significant if 0.5 is outside the interval
        return lower > 0.5 or upper < 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "fine_tuned_agent": self.fine_tuned_agent,
            "baseline_agent": self.baseline_agent,
            "num_trials": self.num_trials,
            "fine_tuned_wins": self.fine_tuned_wins,
            "baseline_wins": self.baseline_wins,
            "draws": self.draws,
            "fine_tuned_win_rate": self.fine_tuned_win_rate,
            "avg_fine_tuned_score": self.avg_fine_tuned_score,
            "avg_baseline_score": self.avg_baseline_score,
            "avg_confidence": self.avg_confidence,
            "consensus_rate": self.consensus_rate,
            "is_significant": self.is_significant,
            "created_at": self.created_at,
        }


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a model."""

    model_id: str
    elo_rating: float
    win_rate: float
    avg_score: float
    calibration_score: float
    consensus_contribution: float
    domain_scores: dict[str, float] = field(default_factory=dict)
    total_debates: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "elo_rating": self.elo_rating,
            "win_rate": self.win_rate,
            "avg_score": self.avg_score,
            "calibration_score": self.calibration_score,
            "consensus_contribution": self.consensus_contribution,
            "domain_scores": self.domain_scores,
            "total_debates": self.total_debates,
        }


class TinkerEvaluator:
    """
    Evaluator for comparing fine-tuned models against baselines.

    Runs controlled A/B tests using the debate Arena and tracks
    performance metrics over time.

    Example:
        evaluator = TinkerEvaluator()

        # Compare fine-tuned vs baseline
        result = await evaluator.a_b_test(
            tasks=["Design a rate limiter", "Optimize database queries"],
            fine_tuned_agent=my_tinker_agent,
            baseline_agent=my_baseline_agent,
            num_trials=10,
        )

        print(f"Fine-tuned win rate: {result.fine_tuned_win_rate:.1%}")
        print(f"Significant: {result.is_significant}")
    """

    def __init__(
        self,
        elo_system: EloSystem | None = None,
        results_dir: Path | str = "evaluation_results",
    ):
        self.elo = elo_system or EloSystem()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._test_counter = 0

    def _generate_test_id(self) -> str:
        """Generate unique test ID."""
        self._test_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"test-{timestamp}-{self._test_counter:04d}"

    async def a_b_test(
        self,
        tasks: list[str],
        fine_tuned_agent: Agent,
        baseline_agent: Agent,
        num_trials: int = 10,
        additional_agents: list[Agent] | None = None,
        randomize_order: bool = True,
        save_results: bool = True,
    ) -> ABTestResult:
        """
        Run A/B test comparing fine-tuned model against baseline.

        Args:
            tasks: List of debate tasks to test on
            fine_tuned_agent: Fine-tuned Tinker agent
            baseline_agent: Baseline agent to compare against
            num_trials: Number of trials per task
            additional_agents: Other agents to include in debates
            randomize_order: Randomize agent order in debates
            save_results: Save results to file

        Returns:
            ABTestResult with comparison statistics
        """
        test_id = self._generate_test_id()
        logger.info(
            "Starting A/B test %s: %s vs %s (%d trials)",
            test_id,
            fine_tuned_agent.name,
            baseline_agent.name,
            num_trials * len(tasks),
        )

        trials = []
        fine_tuned_wins = 0
        baseline_wins = 0
        draws = 0
        total_ft_score = 0.0
        total_bl_score = 0.0
        total_confidence = 0.0
        consensus_count = 0

        for task in tasks:
            for trial_num in range(num_trials):
                # Build agent list
                agents = [fine_tuned_agent, baseline_agent]
                if additional_agents:
                    agents.extend(additional_agents)

                if randomize_order:
                    random.shuffle(agents)

                try:
                    # Run debate
                    debate_result = await self._run_debate(task, agents)

                    # Analyze results - use votes to calculate scores
                    ft_score = 0.0
                    bl_score = 0.0
                    for vote in debate_result.votes:
                        if vote.choice == fine_tuned_agent.name:
                            ft_score += vote.confidence
                        elif vote.choice == baseline_agent.name:
                            bl_score += vote.confidence

                    total_ft_score += ft_score
                    total_bl_score += bl_score
                    total_confidence += debate_result.confidence or 0

                    if debate_result.consensus_reached:
                        consensus_count += 1

                    # Determine winner
                    if ft_score > bl_score:
                        fine_tuned_wins += 1
                        winner = "fine_tuned"
                    elif bl_score > ft_score:
                        baseline_wins += 1
                        winner = "baseline"
                    else:
                        draws += 1
                        winner = "draw"

                    trial_data = {
                        "task": task,
                        "trial": trial_num + 1,
                        "winner": winner,
                        "fine_tuned_score": ft_score,
                        "baseline_score": bl_score,
                        "confidence": debate_result.confidence,
                        "consensus_reached": debate_result.consensus_reached,
                        "rounds_used": debate_result.rounds_used,
                    }
                    trials.append(trial_data)

                    logger.debug(
                        "Trial %d/%d: %s (ft=%.2f, bl=%.2f)",
                        len(trials),
                        num_trials * len(tasks),
                        winner,
                        ft_score,
                        bl_score,
                    )

                except Exception as e:
                    logger.warning("Trial failed: %s", e)
                    trials.append(
                        {
                            "task": task,
                            "trial": trial_num + 1,
                            "error": str(e),
                        }
                    )

        total_trials = fine_tuned_wins + baseline_wins + draws
        win_rate = fine_tuned_wins / total_trials if total_trials > 0 else 0

        result = ABTestResult(
            test_id=test_id,
            fine_tuned_agent=fine_tuned_agent.name,
            baseline_agent=baseline_agent.name,
            num_trials=total_trials,
            fine_tuned_wins=fine_tuned_wins,
            baseline_wins=baseline_wins,
            draws=draws,
            fine_tuned_win_rate=win_rate,
            avg_fine_tuned_score=total_ft_score / total_trials if total_trials > 0 else 0,
            avg_baseline_score=total_bl_score / total_trials if total_trials > 0 else 0,
            avg_confidence=total_confidence / total_trials if total_trials > 0 else 0,
            consensus_rate=consensus_count / total_trials if total_trials > 0 else 0,
            trials=trials,
        )

        if save_results:
            self._save_result(result)

        logger.info(
            "A/B test %s complete: fine_tuned=%d, baseline=%d, draws=%d (win_rate=%.1f%%, significant=%s)",
            test_id,
            fine_tuned_wins,
            baseline_wins,
            draws,
            win_rate * 100,
            result.is_significant,
        )

        return result

    async def _run_debate(
        self,
        task: str,
        agents: list[Agent],
        rounds: int = 3,
    ) -> DebateResult:
        """Run a single debate."""
        from aragora.core import DebateProtocol, Environment
        from aragora.debate.orchestrator import Arena

        env = Environment(task=task)
        protocol = DebateProtocol(
            rounds=rounds,
            consensus="majority",
            early_stop=True,
        )

        arena = Arena(env, agents, protocol)
        return await arena.run()

    def _save_result(self, result: ABTestResult) -> None:
        """Save test result to file."""
        filename = f"{result.test_id}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Also save trials separately for detailed analysis
        trials_file = self.results_dir / f"{result.test_id}_trials.jsonl"
        with open(trials_file, "w") as f:
            for trial in result.trials:
                f.write(json.dumps(trial) + "\n")

    def get_model_metrics(self, agent_name: str) -> EvaluationMetrics:
        """
        Get comprehensive metrics for a model.

        Args:
            agent_name: Name of the agent/model

        Returns:
            EvaluationMetrics with all tracked statistics
        """
        rating = self.elo.get_rating(agent_name)

        # Calculate consensus contribution from debate history
        # (This would need to be tracked during debates)
        consensus_contribution = 0.0

        return EvaluationMetrics(
            model_id=agent_name,
            elo_rating=rating.elo,
            win_rate=rating.win_rate,
            avg_score=0.0,  # Would need score tracking
            calibration_score=rating.calibration_score,
            consensus_contribution=consensus_contribution,
            domain_scores=rating.domain_elos,
            total_debates=rating.games_played,
        )

    async def evaluate_on_benchmark(
        self,
        agent: Agent,
        benchmark_tasks: list[str],
        baseline_agents: list[Agent],
        trials_per_task: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluate an agent on a benchmark task set.

        Args:
            agent: Agent to evaluate
            benchmark_tasks: List of benchmark tasks
            baseline_agents: Agents to compare against
            trials_per_task: Trials per task

        Returns:
            Comprehensive evaluation results
        """
        results: dict[str, Any] = {
            "agent": agent.name,
            "benchmark_size": len(benchmark_tasks),
            "comparisons": [],
            "overall_win_rate": 0.0,
            "overall_elo_gain": 0.0,
        }

        initial_elo = self.elo.get_rating(agent.name).elo
        total_wins = 0
        total_games = 0

        for baseline in baseline_agents:
            ab_result = await self.a_b_test(
                tasks=benchmark_tasks,
                fine_tuned_agent=agent,
                baseline_agent=baseline,
                num_trials=trials_per_task,
            )

            results["comparisons"].append(
                {
                    "baseline": baseline.name,
                    "win_rate": ab_result.fine_tuned_win_rate,
                    "is_significant": ab_result.is_significant,
                }
            )

            total_wins += ab_result.fine_tuned_wins
            total_games += ab_result.num_trials

        final_elo = self.elo.get_rating(agent.name).elo

        results["overall_win_rate"] = total_wins / total_games if total_games > 0 else 0
        results["overall_elo_gain"] = final_elo - initial_elo

        return results

    def load_result(self, test_id: str) -> ABTestResult | None:
        """Load a saved test result."""
        filepath = self.results_dir / f"{test_id}.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        # Load trials
        trials_file = self.results_dir / f"{test_id}_trials.jsonl"
        trials = []
        if trials_file.exists():
            with open(trials_file) as f:
                for line in f:
                    trials.append(json.loads(line))

        return ABTestResult(
            test_id=data["test_id"],
            fine_tuned_agent=data["fine_tuned_agent"],
            baseline_agent=data["baseline_agent"],
            num_trials=data["num_trials"],
            fine_tuned_wins=data["fine_tuned_wins"],
            baseline_wins=data["baseline_wins"],
            draws=data["draws"],
            fine_tuned_win_rate=data["fine_tuned_win_rate"],
            avg_fine_tuned_score=data["avg_fine_tuned_score"],
            avg_baseline_score=data["avg_baseline_score"],
            avg_confidence=data["avg_confidence"],
            consensus_rate=data["consensus_rate"],
            created_at=data.get("created_at", ""),
            trials=trials,
        )

    def list_results(self, limit: int = 50) -> list[dict[str, Any]]:
        """List saved test results."""
        results = []
        for filepath in sorted(self.results_dir.glob("test-*.json"), reverse=True)[:limit]:
            with open(filepath) as f:
                data = json.load(f)
                results.append(
                    {
                        "test_id": data["test_id"],
                        "fine_tuned_agent": data["fine_tuned_agent"],
                        "baseline_agent": data["baseline_agent"],
                        "fine_tuned_win_rate": data["fine_tuned_win_rate"],
                        "is_significant": data.get("is_significant", False),
                        "created_at": data.get("created_at", ""),
                    }
                )
        return results
