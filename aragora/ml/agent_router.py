"""
Agent Router for Task-Based Agent Selection.

Selects optimal agents for specific tasks based on capabilities,
historical performance, and task characteristics.

Usage:
    from aragora.ml import AgentRouter, get_agent_router

    router = get_agent_router()
    decision = router.route(
        task="Implement a rate limiter",
        available_agents=["claude", "gpt-4", "codex", "gemini"],
        team_size=3
    )

    print(decision.selected_agents)  # ["claude", "codex", "gpt-4"]
    print(decision.confidence)  # 0.85

    # Record outcomes for learning
    router.record_performance("claude", task_type="coding", success=True)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set
from collections import defaultdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Categories of tasks for routing."""

    CODING = "coding"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    REASONING = "reasoning"
    RESEARCH = "research"
    MATH = "math"
    GENERAL = "general"


@dataclass
class AgentCapabilities:
    """Known capabilities of an agent."""

    agent_id: str
    strengths: List[TaskType] = field(default_factory=list)
    weaknesses: List[TaskType] = field(default_factory=list)
    speed_tier: int = 2  # 1=fast, 2=medium, 3=slow
    cost_tier: int = 2  # 1=cheap, 2=medium, 3=expensive
    max_context: int = 8000  # Max context window
    supports_code: bool = True
    supports_vision: bool = False
    elo_rating: float = 1000.0

    @classmethod
    def default_capabilities(cls) -> Dict[str, "AgentCapabilities"]:
        """Get default capabilities for known agents."""
        return {
            "claude": cls(
                agent_id="claude",
                strengths=[TaskType.ANALYSIS, TaskType.REASONING, TaskType.CREATIVE],
                speed_tier=2,
                cost_tier=3,
                max_context=200000,
                supports_vision=True,
                elo_rating=1100,
            ),
            "claude-sonnet": cls(
                agent_id="claude-sonnet",
                strengths=[TaskType.CODING, TaskType.ANALYSIS],
                speed_tier=1,
                cost_tier=2,
                max_context=200000,
                supports_vision=True,
                elo_rating=1050,
            ),
            "gpt-4": cls(
                agent_id="gpt-4",
                strengths=[TaskType.REASONING, TaskType.GENERAL],
                speed_tier=2,
                cost_tier=3,
                max_context=128000,
                supports_vision=True,
                elo_rating=1080,
            ),
            "gpt-4o": cls(
                agent_id="gpt-4o",
                strengths=[TaskType.CODING, TaskType.REASONING],
                speed_tier=1,
                cost_tier=2,
                max_context=128000,
                supports_vision=True,
                elo_rating=1070,
            ),
            "codex": cls(
                agent_id="codex",
                strengths=[TaskType.CODING],
                weaknesses=[TaskType.CREATIVE, TaskType.RESEARCH],
                speed_tier=1,
                cost_tier=2,
                max_context=16000,
                elo_rating=1020,
            ),
            "gemini": cls(
                agent_id="gemini",
                strengths=[TaskType.RESEARCH, TaskType.GENERAL],
                speed_tier=2,
                cost_tier=2,
                max_context=1000000,
                supports_vision=True,
                elo_rating=1040,
            ),
            "grok": cls(
                agent_id="grok",
                strengths=[TaskType.CREATIVE, TaskType.GENERAL],
                speed_tier=1,
                cost_tier=2,
                max_context=32000,
                elo_rating=1000,
            ),
            "mistral-large": cls(
                agent_id="mistral-large",
                strengths=[TaskType.CODING, TaskType.REASONING],
                speed_tier=2,
                cost_tier=2,
                max_context=128000,
                elo_rating=1030,
            ),
            "deepseek": cls(
                agent_id="deepseek",
                strengths=[TaskType.CODING, TaskType.MATH],
                speed_tier=2,
                cost_tier=1,
                max_context=64000,
                elo_rating=1010,
            ),
            "llama": cls(
                agent_id="llama",
                strengths=[TaskType.GENERAL],
                speed_tier=1,
                cost_tier=1,
                max_context=8000,
                elo_rating=980,
            ),
        }


@dataclass
class RoutingDecision:
    """Result of agent routing decision."""

    selected_agents: List[str]
    task_type: TaskType
    confidence: float
    reasoning: List[str]
    agent_scores: Dict[str, float] = field(default_factory=dict)
    diversity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_agents": self.selected_agents,
            "task_type": self.task_type.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "diversity_score": round(self.diversity_score, 3),
        }


@dataclass
class AgentRouterConfig:
    """Configuration for agent router."""

    # Scoring weights
    weight_task_match: float = 0.35
    weight_historical: float = 0.25
    weight_elo: float = 0.20
    weight_diversity: float = 0.10
    weight_cost: float = 0.10

    # Constraints
    prefer_diversity: bool = True
    max_same_provider: int = 2
    min_confidence_threshold: float = 0.3

    # Task classification
    use_embeddings: bool = True


class AgentRouter:
    """Routes tasks to optimal agent combinations.

    Uses a combination of:
    1. Task type classification
    2. Agent capability matching
    3. Historical performance
    4. ELO ratings
    5. Team diversity optimization

    Learns from outcomes to improve routing over time.
    """

    def __init__(self, config: Optional[AgentRouterConfig] = None):
        """Initialize the agent router.

        Args:
            config: Router configuration
        """
        self.config = config or AgentRouterConfig()
        self._capabilities = AgentCapabilities.default_capabilities()
        self._embedding_service = None
        self._historical_performance: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._task_type_embeddings: Dict[TaskType, List[float]] = {}

    def _get_embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None and self.config.use_embeddings:
            try:
                from aragora.ml.embeddings import get_embedding_service
                self._embedding_service = get_embedding_service()
            except Exception as e:
                logger.warning(f"Could not load embedding service: {e}")
                self.config.use_embeddings = False
        return self._embedding_service

    def register_agent(self, capabilities: AgentCapabilities) -> None:
        """Register or update agent capabilities.

        Args:
            capabilities: Agent capability profile
        """
        self._capabilities[capabilities.agent_id] = capabilities
        logger.debug(f"Registered agent: {capabilities.agent_id}")

    def _classify_task(self, task: str) -> tuple[TaskType, float]:
        """Classify task type from description.

        Args:
            task: Task description

        Returns:
            Tuple of (task_type, confidence)
        """
        task_lower = task.lower()

        # Keyword-based classification
        patterns = {
            TaskType.CODING: [
                r"\b(code|implement|function|class|api|bug|fix|refactor|test)\b",
                r"\b(python|javascript|java|rust|go|typescript)\b",
                r"\b(algorithm|data structure|optimize)\b",
            ],
            TaskType.MATH: [
                r"\b(calculate|compute|equation|formula|proof|theorem)\b",
                r"\b(math|mathematical|algebra|calculus|statistics)\b",
                r"\b(derivative|integral|probability)\b",
            ],
            TaskType.ANALYSIS: [
                r"\b(analyze|evaluate|compare|assess|review)\b",
                r"\b(pros and cons|tradeoffs|implications)\b",
                r"\b(explain|describe|breakdown)\b",
            ],
            TaskType.CREATIVE: [
                r"\b(write|story|poem|creative|imagine|design)\b",
                r"\b(brainstorm|ideas|suggestions|alternative)\b",
                r"\b(content|marketing|copy)\b",
            ],
            TaskType.REASONING: [
                r"\b(reason|logic|deduce|infer|conclude)\b",
                r"\b(argument|premise|hypothesis)\b",
                r"\b(why|because|therefore|hence)\b",
            ],
            TaskType.RESEARCH: [
                r"\b(research|find|search|look up|investigate)\b",
                r"\b(sources|references|citations|literature)\b",
                r"\b(what is|who is|when did|where)\b",
            ],
        }

        scores = defaultdict(float)
        for task_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, task_lower):
                    scores[task_type] += 1.0

        if not scores:
            return TaskType.GENERAL, 0.5

        # Normalize scores
        total = sum(scores.values())
        for tt in scores:
            scores[tt] /= total

        # Get best match
        best_type = max(scores.keys(), key=lambda x: scores[x])
        confidence = scores[best_type]

        return best_type, min(1.0, confidence + 0.3)  # Base confidence boost

    def _score_agent_for_task(
        self,
        agent_id: str,
        task_type: TaskType,
    ) -> float:
        """Score how well an agent matches a task type.

        Args:
            agent_id: Agent identifier
            task_type: Classified task type

        Returns:
            Match score (0-1)
        """
        capabilities = self._capabilities.get(agent_id)
        if not capabilities:
            return 0.5  # Unknown agent gets neutral score

        score = 0.5  # Base score

        # Strength/weakness adjustment
        if task_type in capabilities.strengths:
            score += 0.3
        if task_type in capabilities.weaknesses:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _get_historical_score(
        self,
        agent_id: str,
        task_type: TaskType,
    ) -> float:
        """Get historical performance score.

        Args:
            agent_id: Agent identifier
            task_type: Task type

        Returns:
            Historical success rate (0-1)
        """
        history = self._historical_performance[agent_id][task_type.value]
        if not history:
            return 0.5  # No history, neutral score

        # Use recent performance (last 50)
        recent = history[-50:]
        return sum(recent) / len(recent)

    def _get_elo_score(self, agent_id: str) -> float:
        """Get normalized ELO score.

        Args:
            agent_id: Agent identifier

        Returns:
            Normalized ELO (0-1)
        """
        capabilities = self._capabilities.get(agent_id)
        if not capabilities:
            return 0.5

        # Normalize ELO: assume range 800-1200
        normalized = (capabilities.elo_rating - 800) / 400
        return max(0.0, min(1.0, normalized))

    def _calculate_diversity_score(
        self,
        selected: List[str],
        all_agents: List[str],
    ) -> float:
        """Calculate diversity score for selected team.

        Args:
            selected: Selected agent IDs
            all_agents: All available agent IDs

        Returns:
            Diversity score (0-1)
        """
        if len(selected) <= 1:
            return 0.0

        # Check strength diversity
        all_strengths: Set[TaskType] = set()
        for agent_id in selected:
            caps = self._capabilities.get(agent_id)
            if caps:
                all_strengths.update(caps.strengths)

        # More diverse strengths = higher score
        strength_diversity = len(all_strengths) / len(TaskType)

        # Check provider diversity (inferred from name)
        providers = set()
        for agent_id in selected:
            if "claude" in agent_id.lower():
                providers.add("anthropic")
            elif "gpt" in agent_id.lower() or "codex" in agent_id.lower():
                providers.add("openai")
            elif "gemini" in agent_id.lower():
                providers.add("google")
            elif "mistral" in agent_id.lower():
                providers.add("mistral")
            else:
                providers.add(agent_id)

        provider_diversity = len(providers) / len(selected)

        return (strength_diversity + provider_diversity) / 2

    def _get_cost_score(self, agent_id: str) -> float:
        """Get cost efficiency score (lower cost = higher score).

        Args:
            agent_id: Agent identifier

        Returns:
            Cost score (0-1, higher is cheaper)
        """
        capabilities = self._capabilities.get(agent_id)
        if not capabilities:
            return 0.5

        # Invert cost tier (1=expensive becomes high score)
        return 1.0 - (capabilities.cost_tier - 1) / 2

    def route(
        self,
        task: str,
        available_agents: Sequence[str],
        team_size: int = 3,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route a task to optimal agents.

        Args:
            task: Task description
            available_agents: List of available agent IDs
            team_size: Desired team size
            constraints: Optional constraints (e.g., max_cost, require_code)

        Returns:
            Routing decision with selected agents
        """
        if not available_agents:
            return RoutingDecision(
                selected_agents=[],
                task_type=TaskType.GENERAL,
                confidence=0.0,
                reasoning=["no_agents_available"],
            )

        constraints = constraints or {}

        # Classify task
        task_type, type_confidence = self._classify_task(task)

        # Score each agent
        agent_scores = {}
        for agent_id in available_agents:
            # Component scores
            task_match = self._score_agent_for_task(agent_id, task_type)
            historical = self._get_historical_score(agent_id, task_type)
            elo = self._get_elo_score(agent_id)
            cost = self._get_cost_score(agent_id)

            # Weighted combination
            score = (
                self.config.weight_task_match * task_match +
                self.config.weight_historical * historical +
                self.config.weight_elo * elo +
                self.config.weight_cost * cost
            )

            # Apply constraints
            caps = self._capabilities.get(agent_id)
            if caps:
                if constraints.get("require_code") and not caps.supports_code:
                    score *= 0.3
                if constraints.get("require_vision") and not caps.supports_vision:
                    score *= 0.3
                if constraints.get("max_cost"):
                    if caps.cost_tier > constraints["max_cost"]:
                        score *= 0.5

            agent_scores[agent_id] = score

        # Select top agents
        sorted_agents = sorted(
            agent_scores.keys(),
            key=lambda x: agent_scores[x],
            reverse=True
        )

        # Optimize for diversity if enabled
        selected = []
        for agent_id in sorted_agents:
            if len(selected) >= team_size:
                break

            # Check diversity constraints
            if self.config.prefer_diversity and selected:
                test_selection = selected + [agent_id]
                diversity = self._calculate_diversity_score(test_selection, list(available_agents))
                if diversity < 0.3 and len(sorted_agents) > team_size:
                    continue  # Skip if would reduce diversity too much

            selected.append(agent_id)

        # Calculate final diversity score
        diversity_score = self._calculate_diversity_score(selected, list(available_agents))

        # Add diversity bonus to overall score
        avg_score = np.mean([agent_scores[a] for a in selected]) if selected else 0.0
        final_confidence = (avg_score + self.config.weight_diversity * diversity_score) * type_confidence

        # Build reasoning
        reasoning = []
        reasoning.append(f"task_type={task_type.value}")
        if selected:
            top_agent = selected[0]
            caps = self._capabilities.get(top_agent)
            if caps and task_type in caps.strengths:
                reasoning.append(f"{top_agent}_strong_at_{task_type.value}")
        if diversity_score > 0.5:
            reasoning.append("good_team_diversity")

        return RoutingDecision(
            selected_agents=selected,
            task_type=task_type,
            confidence=final_confidence,
            reasoning=reasoning,
            agent_scores=agent_scores,
            diversity_score=diversity_score,
        )

    def record_performance(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
    ) -> None:
        """Record agent performance for learning.

        Args:
            agent_id: Agent identifier
            task_type: Type of task performed
            success: Whether task was successful
        """
        self._historical_performance[agent_id][task_type].append(success)

        # Trim history
        history = self._historical_performance[agent_id][task_type]
        if len(history) > 500:
            self._historical_performance[agent_id][task_type] = history[-500:]

        logger.debug(
            f"Recorded performance: {agent_id} on {task_type} = {success}"
        )

    def update_elo(
        self,
        agent_id: str,
        new_elo: float,
    ) -> None:
        """Update agent ELO rating.

        Args:
            agent_id: Agent identifier
            new_elo: New ELO rating
        """
        if agent_id in self._capabilities:
            self._capabilities[agent_id].elo_rating = new_elo
        else:
            # Create default capabilities with updated ELO
            self._capabilities[agent_id] = AgentCapabilities(
                agent_id=agent_id,
                elo_rating=new_elo,
            )

    def get_agent_stats(self, agent_id: str) -> dict[str, Any]:
        """Get statistics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with agent statistics
        """
        caps = self._capabilities.get(agent_id)
        history = self._historical_performance.get(agent_id, {})

        stats = {
            "agent_id": agent_id,
            "registered": caps is not None,
        }

        if caps:
            stats.update({
                "elo_rating": caps.elo_rating,
                "strengths": [s.value for s in caps.strengths],
                "weaknesses": [w.value for w in caps.weaknesses],
                "cost_tier": caps.cost_tier,
                "speed_tier": caps.speed_tier,
            })

        if history:
            total_tasks = sum(len(v) for v in history.values())
            total_success = sum(sum(v) for v in history.values())
            stats["total_tasks"] = total_tasks
            stats["overall_success_rate"] = total_success / total_tasks if total_tasks > 0 else 0.0
            stats["task_breakdown"] = {
                task_type: sum(outcomes) / len(outcomes) if outcomes else 0.0
                for task_type, outcomes in history.items()
            }

        return stats


# Global instance
_agent_router: Optional[AgentRouter] = None


def get_agent_router() -> AgentRouter:
    """Get or create the global agent router instance."""
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter()
    return _agent_router
