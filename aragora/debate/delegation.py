"""
Delegation Strategies for Intelligent Task Routing.

Adapted from ccswarm (MIT License)
Pattern: Intelligent task routing with content-based and load-balanced strategies
Original: https://github.com/nwiizo/ccswarm

Aragora adaptations:
- Integration with ELO-based team selection
- DebateContext workload tracking
- Hybrid strategy combining multiple approaches
- Expertise matching via AgentConfig

Usage:
    # Create a content-based delegator
    delegator = ContentBasedDelegation()
    delegator.add_expertise("security", ["security-auditor", "compliance-auditor"])

    # Select agents for a task
    selected = delegator.select_agents("Find SQL injection vulnerabilities", agents, ctx)

    # Or use hybrid delegation
    hybrid = HybridDelegation([
        (ContentBasedDelegation(), 0.4),
        (LoadBalancedDelegation(), 0.3),
        (ExpertiseDelegation(), 0.3),
    ])
    selected = hybrid.select_agents(task, agents, ctx)
"""

from __future__ import annotations

__all__ = [
    "DelegationStrategy",
    "ContentBasedDelegation",
    "LoadBalancedDelegation",
    "ExpertiseDelegation",
    "HybridDelegation",
    "RoundRobinDelegation",
    "create_default_delegation",
]

import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from aragora.agents.config_loader import AgentConfig
    from aragora.core import Agent
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class DelegationStrategy(ABC):
    """Abstract base class for delegation strategies."""

    @abstractmethod
    def select_agents(
        self,
        task: str,
        agents: Sequence["Agent"],
        context: Optional["DebateContext"] = None,
        max_agents: Optional[int] = None,
    ) -> list["Agent"]:
        """
        Select agents for a task.

        Args:
            task: Task description
            agents: Available agents
            context: Optional debate context with state
            max_agents: Optional maximum number of agents to return

        Returns:
            Ordered list of selected agents (best matches first)
        """
        ...

    @abstractmethod
    def score_agent(
        self,
        agent: "Agent",
        task: str,
        context: Optional["DebateContext"] = None,
    ) -> float:
        """
        Score an individual agent for a task.

        Args:
            agent: Agent to score
            task: Task description
            context: Optional context

        Returns:
            Score (higher is better)
        """
        ...


@dataclass
class ContentBasedDelegation(DelegationStrategy):
    """
    Route tasks by keyword matching to agent expertise.

    Matches keywords in the task description to agent capabilities
    and expertise domains. Agents with more keyword matches score higher.
    """

    keyword_mapping: dict[str, list[str]] = field(default_factory=dict)
    case_sensitive: bool = False
    default_score: float = 1.0

    def add_expertise(self, keyword: str, agent_names: list[str]) -> None:
        """
        Add keyword -> agent mapping.

        Args:
            keyword: Keyword or phrase to match
            agent_names: Agent names that have this expertise
        """
        key = keyword if self.case_sensitive else keyword.lower()
        if key not in self.keyword_mapping:
            self.keyword_mapping[key] = []
        self.keyword_mapping[key].extend(agent_names)

    def add_from_config(self, config: "AgentConfig") -> None:
        """
        Add expertise mappings from an AgentConfig.

        Args:
            config: Agent configuration with expertise_domains and capabilities
        """
        for domain in config.expertise_domains:
            self.add_expertise(domain, [config.name])
        for capability in config.capabilities:
            self.add_expertise(capability, [config.name])

    def _extract_keywords(self, task: str) -> set[str]:
        """Extract keywords from task description."""
        text = task if self.case_sensitive else task.lower()
        # Split on non-alphanumeric characters
        words = set(re.findall(r"\b\w+\b", text))
        return words

    def score_agent(
        self,
        agent: "Agent",
        task: str,
        context: Optional["DebateContext"] = None,
    ) -> float:
        """Score agent based on keyword matches."""
        keywords = self._extract_keywords(task)
        score = self.default_score

        for keyword, agent_names in self.keyword_mapping.items():
            if agent.name in agent_names:
                # Check if keyword is in task
                if keyword in keywords or any(keyword in w for w in keywords):
                    score += 1.0

        return score

    def select_agents(
        self,
        task: str,
        agents: Sequence["Agent"],
        context: Optional["DebateContext"] = None,
        max_agents: Optional[int] = None,
    ) -> list["Agent"]:
        """Select agents by keyword matching."""
        scored = [(a, self.score_agent(a, task, context)) for a in agents]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [a for a, _ in scored]
        if max_agents is not None:
            selected = selected[:max_agents]

        logger.debug(
            f"ContentBasedDelegation selected {len(selected)} agents "
            f"(scores: {[(a.name, s) for a, s in scored[:5]]})"
        )
        return selected


@dataclass
class LoadBalancedDelegation(DelegationStrategy):
    """
    Distribute work evenly across agents.

    Tracks agent workload and prioritizes agents with lower
    current load. Uses context workload tracking if available.
    """

    agent_load: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    max_concurrent_per_agent: int = 3
    default_score: float = 1.0

    def record_task(self, agent_name: str) -> None:
        """Record that an agent received a task."""
        self.agent_load[agent_name] += 1

    def complete_task(self, agent_name: str) -> None:
        """Record that an agent completed a task."""
        if self.agent_load[agent_name] > 0:
            self.agent_load[agent_name] -= 1

    def get_load(self, agent_name: str, context: Optional["DebateContext"] = None) -> int:
        """Get current load for an agent."""
        # Check context for workload data first
        if context is not None and hasattr(context, "agent_workloads"):
            ctx_load = getattr(context, "agent_workloads", {})
            if agent_name in ctx_load:
                return ctx_load[agent_name]

        return self.agent_load.get(agent_name, 0)

    def score_agent(
        self,
        agent: "Agent",
        task: str,
        context: Optional["DebateContext"] = None,
    ) -> float:
        """Score agent inversely proportional to load."""
        load = self.get_load(agent.name, context)

        if load >= self.max_concurrent_per_agent:
            return 0.0  # Agent is overloaded

        # Higher score for lower load
        return self.default_score * (1.0 - load / self.max_concurrent_per_agent)

    def select_agents(
        self,
        task: str,
        agents: Sequence["Agent"],
        context: Optional["DebateContext"] = None,
        max_agents: Optional[int] = None,
    ) -> list["Agent"]:
        """Select agents by load balance."""
        scored = [(a, self.score_agent(a, task, context)) for a in agents]
        # Filter out overloaded agents
        scored = [(a, s) for a, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [a for a, _ in scored]
        if max_agents is not None:
            selected = selected[:max_agents]

        logger.debug(
            f"LoadBalancedDelegation selected {len(selected)} agents "
            f"(loads: {[(a.name, self.get_load(a.name, context)) for a in selected[:5]]})"
        )
        return selected


@dataclass
class ExpertiseDelegation(DelegationStrategy):
    """
    Route based on agent expertise domains.

    Uses AgentConfig expertise_domains to match tasks.
    Requires agents to have _config attribute with AgentConfig.
    """

    domain_keywords: dict[str, list[str]] = field(default_factory=dict)
    default_score: float = 1.0

    def __post_init__(self) -> None:
        """Initialize default domain keywords."""
        if not self.domain_keywords:
            self.domain_keywords = {
                "security": [
                    "security", "vulnerability", "injection", "xss", "credential",
                    "authentication", "authorization", "crypto", "secret", "password",
                ],
                "compliance": [
                    "compliance", "regulatory", "gdpr", "hipaa", "soc2", "pci",
                    "policy", "audit", "governance", "risk",
                ],
                "code-quality": [
                    "quality", "maintainability", "readability", "refactor",
                    "test", "coverage", "documentation", "style", "lint",
                ],
                "performance": [
                    "performance", "optimization", "speed", "latency", "memory",
                    "cpu", "scale", "bottleneck", "profiling",
                ],
            }

    def _get_agent_domains(self, agent: "Agent") -> list[str]:
        """Get expertise domains from agent config."""
        if hasattr(agent, "_config") and agent._config is not None:
            return agent._config.expertise_domains
        return []

    def _match_domains(self, task: str) -> list[str]:
        """Identify domains matching the task."""
        task_lower = task.lower()
        matched = []

        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in task_lower:
                    matched.append(domain)
                    break

        return matched

    def score_agent(
        self,
        agent: "Agent",
        task: str,
        context: Optional["DebateContext"] = None,
    ) -> float:
        """Score agent based on domain match."""
        agent_domains = set(self._get_agent_domains(agent))
        task_domains = set(self._match_domains(task))

        if not task_domains:
            return self.default_score

        overlap = agent_domains & task_domains
        if not overlap:
            return self.default_score

        # Score increases with more domain matches
        return self.default_score + len(overlap)

    def select_agents(
        self,
        task: str,
        agents: Sequence["Agent"],
        context: Optional["DebateContext"] = None,
        max_agents: Optional[int] = None,
    ) -> list["Agent"]:
        """Select agents by expertise match."""
        scored = [(a, self.score_agent(a, task, context)) for a in agents]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [a for a, _ in scored]
        if max_agents is not None:
            selected = selected[:max_agents]

        logger.debug(
            f"ExpertiseDelegation selected {len(selected)} agents "
            f"(matched domains: {self._match_domains(task)})"
        )
        return selected


@dataclass
class RoundRobinDelegation(DelegationStrategy):
    """
    Simple round-robin delegation.

    Rotates through agents in order, ensuring fair distribution
    over time. Uses a cursor to track position.
    """

    cursor: int = 0
    default_score: float = 1.0

    def score_agent(
        self,
        agent: "Agent",
        task: str,
        context: Optional["DebateContext"] = None,
    ) -> float:
        """All agents get equal score in round-robin."""
        return self.default_score

    def select_agents(
        self,
        task: str,
        agents: Sequence["Agent"],
        context: Optional["DebateContext"] = None,
        max_agents: Optional[int] = None,
    ) -> list["Agent"]:
        """Select agents in round-robin order."""
        if not agents:
            return []

        n = len(agents)
        max_select = max_agents if max_agents is not None else n

        # Create ordered list starting from cursor
        selected = []
        for i in range(max_select):
            idx = (self.cursor + i) % n
            selected.append(agents[idx])

        # Advance cursor
        self.cursor = (self.cursor + max_select) % n

        logger.debug(f"RoundRobinDelegation selected {len(selected)} agents (cursor={self.cursor})")
        return selected


@dataclass
class HybridDelegation(DelegationStrategy):
    """
    Combine multiple strategies with weighted priority.

    Aggregates scores from multiple strategies and weights them
    according to specified priorities.
    """

    strategies: list[tuple[DelegationStrategy, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize weights to sum to 1.0."""
        if self.strategies:
            total = sum(w for _, w in self.strategies)
            if total > 0:
                self.strategies = [(s, w / total) for s, w in self.strategies]

    def add_strategy(self, strategy: DelegationStrategy, weight: float) -> None:
        """Add a strategy with its weight."""
        self.strategies.append((strategy, weight))
        # Re-normalize weights
        total = sum(w for _, w in self.strategies)
        if total > 0:
            self.strategies = [(s, w / total) for s, w in self.strategies]

    def score_agent(
        self,
        agent: "Agent",
        task: str,
        context: Optional["DebateContext"] = None,
    ) -> float:
        """Compute weighted score from all strategies."""
        if not self.strategies:
            return 1.0

        total_score = 0.0
        for strategy, weight in self.strategies:
            score = strategy.score_agent(agent, task, context)
            total_score += score * weight

        return total_score

    def select_agents(
        self,
        task: str,
        agents: Sequence["Agent"],
        context: Optional["DebateContext"] = None,
        max_agents: Optional[int] = None,
    ) -> list["Agent"]:
        """Select agents using weighted combined scoring."""
        scored = [(a, self.score_agent(a, task, context)) for a in agents]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [a for a, _ in scored]
        if max_agents is not None:
            selected = selected[:max_agents]

        logger.debug(
            f"HybridDelegation selected {len(selected)} agents "
            f"using {len(self.strategies)} strategies"
        )
        return selected


def create_default_delegation(
    include_content: bool = True,
    include_load: bool = True,
    include_expertise: bool = True,
    content_weight: float = 0.4,
    load_weight: float = 0.3,
    expertise_weight: float = 0.3,
) -> HybridDelegation:
    """
    Create a default hybrid delegation strategy.

    Args:
        include_content: Include content-based delegation
        include_load: Include load-balanced delegation
        include_expertise: Include expertise-based delegation
        content_weight: Weight for content-based strategy
        load_weight: Weight for load-balanced strategy
        expertise_weight: Weight for expertise-based strategy

    Returns:
        Configured HybridDelegation instance
    """
    strategies: list[tuple[DelegationStrategy, float]] = []

    if include_content:
        content = ContentBasedDelegation()
        # Add default security keywords
        content.add_expertise("security", ["security-auditor"])
        content.add_expertise("vulnerability", ["security-auditor"])
        content.add_expertise("compliance", ["compliance-auditor"])
        content.add_expertise("quality", ["quality-reviewer"])
        strategies.append((content, content_weight))

    if include_load:
        strategies.append((LoadBalancedDelegation(), load_weight))

    if include_expertise:
        strategies.append((ExpertiseDelegation(), expertise_weight))

    return HybridDelegation(strategies=strategies)
