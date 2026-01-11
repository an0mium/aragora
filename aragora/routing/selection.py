"""
Adaptive Agent Selection using ELO and Personas.

Routes tasks to best-fit agents by:
- Matching task domain to agent expertise
- Using ELO ratings for quality ranking
- Using probe vulnerability scores for reliability
- Forming optimal teams for debates
- Maintaining a "bench" system with promotion/demotion
- Auto-detecting domain from task text
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
import random
import math

logger = logging.getLogger(__name__)


# Domain detection keywords
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "security": [
        "security", "auth", "authentication", "authorization", "encrypt",
        "vulnerability", "attack", "xss", "sql injection", "csrf", "token",
        "password", "credential", "permission", "access control", "firewall",
        "sanitize", "validate input", "owasp", "rate limit", "brute force",
    ],
    "performance": [
        "performance", "optimize", "speed", "latency", "cache", "caching",
        "memory", "cpu", "throughput", "bottleneck", "profil", "benchmark",
        "slow", "fast", "efficient", "scale", "scaling", "load", "concurrent",
    ],
    "architecture": [
        "architecture", "design", "pattern", "refactor", "structure",
        "modular", "decouple", "interface", "abstract", "dependency",
        "solid", "dry", "single responsibility", "microservice", "monolith",
    ],
    "testing": [
        "test", "testing", "unittest", "pytest", "mock", "coverage",
        "integration test", "e2e", "end-to-end", "tdd", "bdd", "fixture",
        "assertion", "spec", "verify", "validate",
    ],
    "api": [
        "api", "endpoint", "rest", "graphql", "grpc", "http", "request",
        "response", "route", "handler", "middleware", "cors", "versioning",
        "openapi", "swagger", "webhook",
    ],
    "database": [
        "database", "db", "sql", "query", "schema", "migration", "index",
        "transaction", "orm", "postgresql", "mysql", "sqlite", "mongodb",
        "redis", "cache", "nosql", "join", "foreign key",
    ],
    "frontend": [
        "frontend", "ui", "ux", "react", "vue", "angular", "css", "html",
        "component", "render", "state", "redux", "hook", "responsive",
        "accessibility", "a11y", "animation",
    ],
    "devops": [
        "deploy", "deployment", "ci", "cd", "docker", "kubernetes", "k8s",
        "pipeline", "github actions", "terraform", "aws", "cloud", "infra",
        "monitoring", "logging", "observability",
    ],
    "debugging": [
        "debug", "bug", "fix", "error", "exception", "traceback", "crash",
        "issue", "problem", "broken", "fail", "not working", "investigate",
    ],
    "documentation": [
        "document", "readme", "docstring", "comment", "explain", "tutorial",
        "guide", "specification", "api doc",
    ],
    "ethics": [
        "ethics", "ethical", "fairness", "bias", "privacy", "consent",
        "responsible", "governance", "compliance", "gdpr", "moral",
        "transparency", "accountability", "harm", "safety", "alignment",
    ],
    "philosophy": [
        "philosophy", "philosophical", "epistemology", "epistemological",
        "ontology", "ontological", "logic", "logical", "reasoning",
        "argument", "premise", "conclusion", "fallacy", "dialectic",
        "metaphysics", "metaphysical", "theory of", "concept", "definition",
        "truth claim", "knowledge", "belief", "justify", "justification",
        "foundational", "first principles",
    ],
    "data_analysis": [
        "data", "analysis", "dataset", "statistics", "statistical", "pandas",
        "numpy", "visualization", "chart", "plot", "correlation", "regression",
        "machine learning", "ml", "prediction", "model", "feature", "training",
        "jupyter", "notebook", "csv", "json", "etl", "pipeline",
    ],
    "general": [
        "implement", "create", "build", "add", "update", "change", "modify",
        "code", "function", "class", "method", "module", "library", "package",
    ],
}


# Default agent expertise profiles
DEFAULT_AGENT_EXPERTISE: dict[str, dict[str, float]] = {
    "claude": {
        "security": 0.9,
        "architecture": 0.9,
        "documentation": 0.95,
        "api": 0.85,
        "debugging": 0.85,
        "testing": 0.8,
        "frontend": 0.75,
        "database": 0.8,
        "ethics": 0.95,  # Claude excels at ethical reasoning
        "philosophy": 0.9,  # Strong philosophical analysis
        "data_analysis": 0.8,
        "general": 0.85,
    },
    "codex": {
        "performance": 0.9,
        "debugging": 0.9,
        "testing": 0.85,
        "api": 0.85,
        "database": 0.85,
        "architecture": 0.8,
        "devops": 0.75,
        "data_analysis": 0.85,  # Strong at data pipelines
        "general": 0.9,  # Primary implementer
    },
    "gemini": {
        "architecture": 0.95,
        "performance": 0.85,
        "api": 0.85,
        "documentation": 0.85,
        "frontend": 0.8,
        "database": 0.8,
        "data_analysis": 0.9,  # Excellent at data analysis
        "philosophy": 0.8,
        "general": 0.85,
    },
    "grok": {
        "debugging": 0.9,
        "security": 0.85,
        "testing": 0.85,
        "performance": 0.8,
        "architecture": 0.75,
        "philosophy": 0.85,  # Good at lateral thinking
        "ethics": 0.75,
        "general": 0.8,
    },
    "deepseek": {
        "architecture": 0.9,
        "performance": 0.9,
        "database": 0.85,
        "api": 0.85,
        "security": 0.8,
        "data_analysis": 0.9,  # Strong at analysis
        "philosophy": 0.85,  # Rigorous reasoning
        "general": 0.85,
    },
}


class DomainDetector:
    """
    Detects task domain from natural language description.

    Uses keyword matching with weighted scoring to identify
    the primary and secondary domains for a task.
    """

    def __init__(self, custom_keywords: Optional[dict[str, list[str]]] = None):
        """Initialize with optional custom domain keywords."""
        self.keywords = DOMAIN_KEYWORDS.copy()
        if custom_keywords:
            for domain, words in custom_keywords.items():
                if domain in self.keywords:
                    self.keywords[domain].extend(words)
                else:
                    self.keywords[domain] = words

    def detect(self, task_text: str, top_n: int = 3) -> list[tuple[str, float]]:
        """
        Detect domains from task text.

        Args:
            task_text: The task description
            top_n: Number of top domains to return

        Returns:
            List of (domain, confidence) tuples, sorted by confidence
        """
        text_lower = task_text.lower()
        scores: dict[str, float] = {}

        for domain, keywords in self.keywords.items():
            score = 0.0
            for keyword in keywords:
                # Count occurrences with word boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    # Longer keywords are more specific, weight them higher
                    weight = 1.0 + len(keyword.split()) * 0.5
                    score += matches * weight

            if score > 0:
                scores[domain] = score

        # Normalize scores
        if scores:
            max_score = max(scores.values())
            scores = {d: s / max_score for d, s in scores.items()}

        # Sort by score descending
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Default to "general" if no domains detected
        if not sorted_domains:
            return [("general", 0.5)]

        return sorted_domains[:top_n]

    def get_primary_domain(self, task_text: str) -> str:
        """Get the primary domain for a task."""
        domains = self.detect(task_text, top_n=1)
        return domains[0][0] if domains else "general"

    def get_task_requirements(
        self,
        task_text: str,
        task_id: Optional[str] = None,
    ) -> "TaskRequirements":
        """
        Create TaskRequirements from task text with auto-detected domains.

        Args:
            task_text: The task description
            task_id: Optional task identifier

        Returns:
            TaskRequirements with detected domains
        """
        domains = self.detect(task_text, top_n=3)

        primary = domains[0][0] if domains else "general"
        secondary = [d for d, _ in domains[1:] if _ > 0.3]  # Only include confident secondary

        # Detect required traits from keywords
        traits = []
        text_lower = task_text.lower()
        if any(w in text_lower for w in ["critical", "important", "careful"]):
            traits.append("thorough")
        if any(w in text_lower for w in ["fast", "quick", "asap"]):
            traits.append("fast")
        if any(w in text_lower for w in ["security", "secure", "safe"]):
            traits.append("security")
        if any(w in text_lower for w in ["creative", "novel", "innovative"]):
            traits.append("creative")

        return TaskRequirements(
            task_id=task_id or f"task-{hash(task_text) % 10000:04d}",
            description=task_text[:500],
            primary_domain=primary,
            secondary_domains=secondary,
            required_traits=traits,
        )

if TYPE_CHECKING:
    from aragora.routing.probe_filter import ProbeFilter


@dataclass
class AgentProfile:
    """Profile of an available agent."""

    name: str
    agent_type: str  # "claude", "codex", "gemini", etc.
    elo_rating: float = 1500
    domain_ratings: dict[str, float] = field(default_factory=dict)
    expertise: dict[str, float] = field(default_factory=dict)  # domain -> 0-1
    traits: list[str] = field(default_factory=list)
    availability: float = 1.0  # 0-1, how available is this agent
    cost_factor: float = 1.0  # Relative cost multiplier
    latency_ms: float = 1000  # Average response latency
    success_rate: float = 0.8  # Historical success rate

    # Probe-based reliability metrics (from ProbeFilter)
    probe_score: float = 1.0  # 1 = no vulnerabilities, 0 = high vulnerability
    has_critical_probes: bool = False  # True if agent has critical probe failures

    # Calibration metrics (from CalibrationTracker)
    calibration_score: float = 1.0  # 1 = well-calibrated, 0 = poorly calibrated
    brier_score: float = 0.0  # Lower is better (0 = perfect predictions)
    is_overconfident: bool = False  # True if agent's confidence exceeds accuracy

    @property
    def overall_score(self) -> float:
        """Calculate overall agent quality score, including probe and calibration."""
        # Base score composition:
        # - 30% ELO rating
        # - 20% success rate
        # - 20% probe reliability
        # - 15% calibration quality
        # - 15% speed/cost
        base_score = (
            self.elo_rating / 2000 * 0.30 +
            self.success_rate * 0.20 +
            self.probe_score * 0.20 +
            self.calibration_score * 0.15 +
            (1 - min(self.latency_ms, 5000) / 5000) * 0.075 +
            (1 - min(self.cost_factor, 3) / 3) * 0.075
        )
        # Penalties for critical issues
        if self.has_critical_probes:
            base_score *= 0.7
        if self.is_overconfident:
            base_score *= 0.9  # 10% penalty for overconfidence
        return base_score


@dataclass
class TaskRequirements:
    """Requirements for a task."""

    task_id: str
    description: str
    primary_domain: str
    secondary_domains: list[str] = field(default_factory=list)
    required_traits: list[str] = field(default_factory=list)
    min_agents: int = 2
    max_agents: int = 5
    quality_priority: float = 0.5  # 0 = speed/cost, 1 = quality
    diversity_preference: float = 0.5  # 0 = homogeneous, 1 = diverse


@dataclass
class TeamComposition:
    """A selected team of agents."""

    team_id: str
    task_id: str
    agents: list[AgentProfile]
    roles: dict[str, str]  # agent_name -> role
    expected_quality: float
    expected_cost: float
    diversity_score: float
    rationale: str


class AgentSelector:
    """
    Selects optimal agents for tasks based on ELO, expertise, and team dynamics.

    Features:
    - Domain-aware selection
    - Probe-aware reliability weighting
    - Calibration-aware confidence weighting
    - Team diversity optimization
    - Cost/quality tradeoffs
    - Bench system for testing new agents
    """

    def __init__(
        self,
        elo_system: Optional[Any] = None,
        persona_manager: Optional[Any] = None,
        probe_filter: Optional["ProbeFilter"] = None,
        calibration_tracker: Optional[Any] = None,
        performance_monitor: Optional[Any] = None,
    ):
        self.elo_system = elo_system
        self.persona_manager = persona_manager
        self.probe_filter = probe_filter
        self.calibration_tracker = calibration_tracker
        self.performance_monitor = performance_monitor
        self.agent_pool: dict[str, AgentProfile] = {}
        self.bench: list[str] = []  # Agents on the bench (probation/testing)
        self._selection_history: list[dict] = []
        # Cached performance insights (refreshed before selection)
        self._performance_insights: dict = {}

    def register_agent(self, profile: AgentProfile):
        """Register an agent in the pool."""
        self.agent_pool[profile.name] = profile

    def remove_agent(self, name: str):
        """Remove an agent from the pool."""
        if name in self.agent_pool:
            del self.agent_pool[name]
        if name in self.bench:
            self.bench.remove(name)

    def move_to_bench(self, name: str):
        """Move an agent to the bench (probation)."""
        if name in self.agent_pool and name not in self.bench:
            self.bench.append(name)

    def promote_from_bench(self, name: str):
        """Promote an agent from bench to active pool."""
        if name in self.bench:
            self.bench.remove(name)

    def set_probe_filter(self, probe_filter: "ProbeFilter"):
        """Set or update the probe filter for reliability scoring."""
        self.probe_filter = probe_filter
        # Refresh scores immediately
        self.refresh_probe_scores()

    def refresh_probe_scores(self):
        """
        Refresh probe scores for all agents in the pool.

        Queries the ProbeFilter for each agent's vulnerability profile
        and updates their probe_score and has_critical_probes fields.
        Call this periodically or before team selection.
        """
        if not self.probe_filter:
            return

        for agent_name, profile in self.agent_pool.items():
            try:
                probe_profile = self.probe_filter.get_agent_profile(agent_name)
                profile.probe_score = probe_profile.probe_score
                profile.has_critical_probes = probe_profile.has_critical_issues()
            except Exception as e:
                # Agent not in probe system - use defaults
                logger.debug(f"Failed to sync probe profile for {agent_name}: {e}. Using defaults.")
                profile.probe_score = 1.0
                profile.has_critical_probes = False

    def refresh_from_elo_system(self, elo_system: Optional[Any] = None):
        """
        Sync domain ratings from EloSystem for all agents in the pool.

        Updates each agent's elo_rating, domain_ratings, and success_rate
        from the EloSystem's current state. Call this periodically or
        before team selection to ensure ratings are current.

        Args:
            elo_system: EloSystem instance to sync from (uses self.elo_system if None)
        """
        elo = elo_system or self.elo_system
        if not elo:
            return

        # Batch fetch all ratings in single query
        agent_names = list(self.agent_pool.keys())
        ratings = elo.get_ratings_batch(agent_names)

        for agent_name, profile in self.agent_pool.items():
            rating = ratings.get(agent_name)
            if rating:
                profile.elo_rating = rating.elo
                profile.domain_ratings = rating.domain_elos.copy() if rating.domain_elos else {}
                profile.success_rate = rating.win_rate

    def get_probe_adjusted_score(self, agent_name: str, base_score: float) -> float:
        """
        Adjust a score based on agent's probe reliability.

        Args:
            agent_name: Name of the agent
            base_score: The base score to adjust

        Returns:
            Adjusted score (base_score * probe_reliability_factor)
        """
        probe_score = 1.0
        has_critical = False

        # First, check if we have probe data from the filter
        if self.probe_filter:
            try:
                probe_profile = self.probe_filter.get_agent_profile(agent_name)
                if probe_profile.total_probes > 0:  # Only use if agent has been probed
                    probe_score = probe_profile.probe_score
                    has_critical = probe_profile.has_critical_issues()
            except Exception as e:
                logger.debug(f"Probe lookup failed for {agent_name}: {e}")

        # Fall back to agent profile's probe_score if no filter data
        if probe_score == 1.0 and agent_name in self.agent_pool:
            agent = self.agent_pool[agent_name]
            probe_score = agent.probe_score
            has_critical = agent.has_critical_probes

        # Apply adjustment: range 0.5 (vulnerable) to 1.0 (reliable)
        adjustment = 0.5 + (probe_score * 0.5)

        # Extra penalty for critical issues
        if has_critical:
            adjustment *= 0.8

        return base_score * adjustment

    def set_calibration_tracker(self, calibration_tracker: Any):
        """Set or update the calibration tracker for confidence scoring."""
        self.calibration_tracker = calibration_tracker
        # Refresh scores immediately
        self.refresh_calibration_scores()

    def refresh_calibration_scores(self):
        """
        Refresh calibration scores for all agents in the pool.

        Queries the CalibrationTracker for each agent's calibration metrics
        and updates their calibration_score, brier_score, and is_overconfident fields.
        """
        if not self.calibration_tracker:
            return

        for agent_name, profile in self.agent_pool.items():
            try:
                summary = self.calibration_tracker.get_calibration_summary(agent_name)
                if summary.total_predictions >= 5:  # Need enough data
                    # calibration_score = 1 - ECE (lower ECE = better calibration)
                    profile.calibration_score = max(0.0, 1.0 - summary.ece)
                    profile.brier_score = summary.brier_score
                    profile.is_overconfident = summary.is_overconfident
                else:
                    # Not enough data - use defaults
                    profile.calibration_score = 1.0
                    profile.brier_score = 0.0
                    profile.is_overconfident = False
            except Exception as e:
                # Agent not in calibration system - use defaults
                logger.debug(f"Failed to sync calibration for {agent_name}: {e}. Using defaults.")
                profile.calibration_score = 1.0
                profile.brier_score = 0.0
                profile.is_overconfident = False

    def get_calibration_adjusted_score(self, agent_name: str, base_score: float) -> float:
        """
        Adjust a score based on agent's calibration quality.

        Args:
            agent_name: Name of the agent
            base_score: The base score to adjust

        Returns:
            Adjusted score (base_score * calibration_factor)
        """
        calibration_score = 1.0
        is_overconfident = False

        # Check calibration tracker first
        if self.calibration_tracker:
            try:
                summary = self.calibration_tracker.get_calibration_summary(agent_name)
                if summary.total_predictions >= 5:
                    calibration_score = max(0.0, 1.0 - summary.ece)
                    is_overconfident = summary.is_overconfident
            except Exception as e:
                logger.debug(f"Calibration lookup failed for {agent_name}: {e}")

        # Fall back to agent profile if no tracker data
        if calibration_score == 1.0 and agent_name in self.agent_pool:
            agent = self.agent_pool[agent_name]
            calibration_score = agent.calibration_score
            is_overconfident = agent.is_overconfident

        # Apply adjustment: range 0.7 (poorly calibrated) to 1.0 (well calibrated)
        adjustment = 0.7 + (calibration_score * 0.3)

        # Penalty for overconfidence
        if is_overconfident:
            adjustment *= 0.9

        return base_score * adjustment

    def set_performance_monitor(self, performance_monitor: Any):
        """Set or update the performance monitor for reliability scoring."""
        self.performance_monitor = performance_monitor
        self.refresh_performance_insights()

    def refresh_performance_insights(self):
        """
        Refresh performance insights from the PerformanceMonitor.

        Call this before team selection to get current performance data.
        Updates internal cache with latest success rates, timeouts, etc.
        """
        if not self.performance_monitor:
            self._performance_insights = {}
            return

        try:
            self._performance_insights = self.performance_monitor.get_performance_insights()
            logger.debug(f"Refreshed performance insights for {len(self._performance_insights.get('agent_stats', {}))} agents")
        except Exception as e:
            logger.warning(f"Failed to refresh performance insights: {e}")
            self._performance_insights = {}

    def get_performance_adjusted_score(self, agent_name: str, base_score: float) -> float:
        """
        Adjust a score based on agent's performance metrics.

        Penalizes agents with:
        - Low success rate (<70%): 20% penalty
        - High timeout rate (>20%): 30% penalty
        - High failure rate (>30%): 25% penalty

        Args:
            agent_name: Name of the agent
            base_score: The base score to adjust

        Returns:
            Adjusted score (may be reduced for unreliable agents)
        """
        if not self._performance_insights:
            return base_score

        agent_stats = self._performance_insights.get("agent_stats", {}).get(agent_name, {})
        if not agent_stats:
            return base_score

        adjustment = 1.0

        # Penalize low success rate
        success_rate = agent_stats.get("success_rate", 100)
        if success_rate < 70:
            adjustment *= 0.8  # 20% penalty
            logger.debug(f"Agent {agent_name} penalized for low success rate: {success_rate:.1f}%")
        elif success_rate < 85:
            adjustment *= 0.9  # 10% penalty for moderate issues

        # Penalize timeout-prone agents
        timeout_rate = agent_stats.get("timeout_rate", 0)
        if timeout_rate > 20:
            adjustment *= 0.7  # 30% penalty
            logger.debug(f"Agent {agent_name} penalized for high timeout rate: {timeout_rate:.1f}%")
        elif timeout_rate > 10:
            adjustment *= 0.85  # 15% penalty

        # Penalize high failure rate
        failure_rate = agent_stats.get("failure_rate", 0)
        if failure_rate > 30:
            adjustment *= 0.75  # 25% penalty
            logger.debug(f"Agent {agent_name} penalized for high failure rate: {failure_rate:.1f}%")

        return base_score * adjustment

    def select_team(
        self,
        requirements: TaskRequirements,
        exclude: Optional[list[str]] = None,
    ) -> TeamComposition:
        """
        Select an optimal team for the task.

        Args:
            requirements: Task requirements
            exclude: Agent names to exclude

        Returns:
            TeamComposition with selected agents
        """
        exclude = exclude or []

        # Refresh performance insights before scoring
        self.refresh_performance_insights()

        candidates = [
            a for a in self.agent_pool.values()
            if a.name not in exclude and a.name not in self.bench
        ]

        if not candidates:
            raise ValueError("No available agents in pool")

        # Score candidates for this task
        scored = [(a, self._score_for_task(a, requirements)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select team considering diversity
        team = self._select_diverse_team(
            scored,
            requirements.min_agents,
            requirements.max_agents,
            requirements.diversity_preference,
        )

        # Assign roles
        roles = self._assign_roles(team, requirements)

        # Calculate expected quality
        expected_quality = sum(
            self._score_for_task(a, requirements) for a in team
        ) / len(team)

        # Calculate cost
        expected_cost = sum(a.cost_factor for a in team)

        # Calculate diversity
        diversity_score = self._calculate_diversity(team)

        # Generate rationale
        rationale = self._generate_rationale(team, requirements, scored)

        # Record selection
        self._selection_history.append({
            "task_id": requirements.task_id,
            "selected": [a.name for a in team],
            "timestamp": datetime.now().isoformat(),
        })

        return TeamComposition(
            team_id=f"team-{requirements.task_id}",
            task_id=requirements.task_id,
            agents=team,
            roles=roles,
            expected_quality=expected_quality,
            expected_cost=expected_cost,
            diversity_score=diversity_score,
            rationale=rationale,
        )

    def _score_for_task(
        self,
        agent: AgentProfile,
        requirements: TaskRequirements,
    ) -> float:
        """Score an agent for a specific task."""
        score = 0.0

        # Base ELO contribution
        elo_score = (agent.elo_rating - 1000) / 1000  # Normalize around 0
        score += elo_score * 0.3

        # Get dynamic expertise from PersonaManager if available
        expertise = agent.expertise.copy()
        traits = agent.traits.copy() if agent.traits else []
        if self.persona_manager:
            try:
                persona = self.persona_manager.get_persona(agent.name)
                if persona:
                    # Merge learned expertise (overrides static expertise)
                    for domain, exp_score in persona.expertise.items():
                        expertise[domain] = exp_score
                    # Merge learned traits
                    for trait in persona.traits:
                        if trait not in traits:
                            traits.append(trait)
            except Exception as e:
                logger.debug(f"Failed to get persona for {agent.name}: {e}. Using static expertise.")

        # Domain expertise (using dynamic expertise)
        primary_exp = expertise.get(requirements.primary_domain, 0.5)
        score += primary_exp * 0.3

        # Domain-specific ELO
        domain_elo = agent.domain_ratings.get(requirements.primary_domain, 1500)
        score += (domain_elo - 1000) / 1000 * 0.2

        # Secondary domains (using dynamic expertise)
        if requirements.secondary_domains:
            secondary_score = sum(
                expertise.get(d, 0.3)
                for d in requirements.secondary_domains
            ) / len(requirements.secondary_domains)
            score += secondary_score * 0.1

        # Trait matching (using dynamic traits)
        if requirements.required_traits:
            matching_traits = sum(
                1 for t in requirements.required_traits
                if t in traits
            )
            score += matching_traits / len(requirements.required_traits) * 0.1

        # Adjust for quality priority
        if requirements.quality_priority > 0.5:
            # Prefer quality: weight success rate more
            score = score * 0.7 + agent.success_rate * 0.3
        else:
            # Prefer speed/cost: weight latency/cost more
            speed_score = 1 - min(agent.latency_ms, 5000) / 5000
            cost_score = 1 - min(agent.cost_factor, 3) / 3
            score = score * 0.6 + speed_score * 0.2 + cost_score * 0.2

        # Apply probe reliability adjustment
        # Agents with higher vulnerability rates get penalized
        score = self.get_probe_adjusted_score(agent.name, score)

        # Apply calibration adjustment
        # Agents with poor calibration (overconfident) get penalized
        score = self.get_calibration_adjusted_score(agent.name, score)

        # Apply performance-based adjustment
        # Agents with low success rates, high timeouts, or failures get penalized
        score = self.get_performance_adjusted_score(agent.name, score)

        return max(0, min(1, score))

    def _select_diverse_team(
        self,
        scored: list[tuple[AgentProfile, float]],
        min_size: int,
        max_size: int,
        diversity_pref: float,
    ) -> list[AgentProfile]:
        """Select a diverse team from scored candidates."""
        if len(scored) <= min_size:
            return [a for a, _ in scored]

        team: list[AgentProfile] = []
        remaining = list(scored)

        while len(team) < max_size and remaining:
            if len(team) < min_size or random.random() > diversity_pref:
                # Greedy: pick highest scored
                agent, score = remaining[0]
                team.append(agent)
                remaining = remaining[1:]
            else:
                # Diversity: pick someone different
                team_types = set(a.agent_type for a in team)
                team_traits = set()
                for a in team:
                    team_traits.update(a.traits)

                # Find most different agent
                best_diff: AgentProfile | None = None
                best_diff_score: float = -1.0

                for agent, score in remaining:
                    diff_score: float = 0.0
                    if agent.agent_type not in team_types:
                        diff_score += 0.5
                    new_traits = set(agent.traits) - team_traits
                    diff_score += len(new_traits) * 0.1
                    diff_score += score * 0.4  # Still consider quality

                    if diff_score > best_diff_score:
                        best_diff = agent
                        best_diff_score = diff_score

                if best_diff:
                    team.append(best_diff)
                    remaining = [(a, s) for a, s in remaining if a.name != best_diff.name]
                else:
                    break

        return team

    def _assign_roles(
        self,
        team: list[AgentProfile],
        requirements: TaskRequirements,
    ) -> dict[str, str]:
        """Assign debate roles to team members."""
        roles: dict[str, str] = {}

        if not team:
            return roles

        # Sort by domain expertise for proposer selection
        by_expertise = sorted(
            team,
            key=lambda a: a.expertise.get(requirements.primary_domain, 0),
            reverse=True,
        )

        # Assign proposer to highest domain expert
        roles[by_expertise[0].name] = "proposer"

        # Assign synthesizer to most balanced agent
        if len(team) > 1:
            remaining = by_expertise[1:]
            balanced = min(
                remaining,
                key=lambda a: abs(a.overall_score - 0.5),
            )
            roles[balanced.name] = "synthesizer"

        # Assign critics to rest
        for agent in team:
            if agent.name not in roles:
                # Try to match critic type to traits
                if "thorough" in agent.traits or "security" in agent.traits:
                    roles[agent.name] = "security_critic"
                elif "pragmatic" in agent.traits or "performance" in agent.traits:
                    roles[agent.name] = "performance_critic"
                else:
                    roles[agent.name] = "critic"

        return roles

    def assign_hybrid_roles(
        self,
        team: list[AgentProfile],
        phase: str,
    ) -> dict[str, str]:
        """
        Assign phase-specific roles for Hybrid Model Architecture.

        Architecture:
        - Gemini: Primary planner/designer (leads Phase 2)
        - Claude: Primary implementer (leads Phase 3)
        - Codex: Primary verifier (leads Phase 4)
        - Grok: Lateral thinker/devil's advocate (critiques all phases)
        - DeepSeek: Rigorous analyst/formal reasoner (validates logic all phases)
        """
        roles = {}
        agent_map = {a.name.lower(): a for a in team}

        # Helper to find agent by type
        def find_agent(agent_type: str) -> AgentProfile | None:
            for agent in team:
                if agent_type in agent.name.lower() or agent.agent_type == agent_type:
                    return agent
            return None

        gemini = find_agent("gemini")
        claude = find_agent("claude")
        codex = find_agent("codex")
        grok = find_agent("grok")
        deepseek = find_agent("deepseek")

        if phase == "debate":
            # All agents are proposers in debate phase
            for agent in team:
                roles[agent.name] = "proposer"

        elif phase == "design":
            # Gemini leads design as primary proposer
            if gemini:
                roles[gemini.name] = "design_lead"
            if claude:
                roles[claude.name] = "architecture_critic"
            if codex:
                roles[codex.name] = "implementation_critic"
            if grok:
                roles[grok.name] = "devil_advocate"
            if deepseek:
                roles[deepseek.name] = "logic_validator"
            # Fallback for any unassigned
            for agent in team:
                if agent.name not in roles:
                    roles[agent.name] = "critic"

        elif phase == "implement":
            # Claude leads implementation
            if claude:
                roles[claude.name] = "implementer"
            # Others are advisors
            for agent in team:
                if agent.name not in roles:
                    roles[agent.name] = "advisor"

        elif phase == "verify":
            # Codex leads verification
            if codex:
                roles[codex.name] = "verification_lead"
            if grok:
                roles[grok.name] = "quality_auditor"
            if gemini:
                roles[gemini.name] = "design_validator"
            if claude:
                roles[claude.name] = "implementation_reviewer"
            if deepseek:
                roles[deepseek.name] = "formal_verifier"
            # Fallback
            for agent in team:
                if agent.name not in roles:
                    roles[agent.name] = "reviewer"

        elif phase == "commit":
            # All agents review commit
            for agent in team:
                roles[agent.name] = "reviewer"

        else:
            # Fallback to standard role assignment
            for agent in team:
                roles[agent.name] = "participant"

        return roles

    def _calculate_diversity(self, team: list[AgentProfile]) -> float:
        """Calculate team diversity score."""
        if len(team) <= 1:
            return 0.0

        # Type diversity
        types = set(a.agent_type for a in team)
        type_div = len(types) / len(team)

        # Trait diversity
        all_traits = set()
        for a in team:
            all_traits.update(a.traits)
        trait_div = len(all_traits) / (len(team) * 3)  # Assume avg 3 traits

        # ELO diversity (mix of skill levels)
        elos = [a.elo_rating for a in team]
        elo_range = max(elos) - min(elos) if len(elos) > 1 else 0
        elo_div = min(elo_range / 500, 1.0)  # Normalize to 500 range

        return (type_div * 0.4 + trait_div * 0.3 + elo_div * 0.3)

    def _generate_rationale(
        self,
        team: list[AgentProfile],
        requirements: TaskRequirements,
        all_scored: list[tuple[AgentProfile, float]],
    ) -> str:
        """Generate human-readable selection rationale."""
        lines = [
            f"Selected {len(team)} agents for task '{requirements.task_id}':",
            f"Primary domain: {requirements.primary_domain}",
            "",
            "Team composition:",
        ]

        for agent in team:
            expertise = agent.expertise.get(requirements.primary_domain, 0)
            domain_elo = agent.domain_ratings.get(requirements.primary_domain, agent.elo_rating)
            lines.append(
                f"- {agent.name} ({agent.agent_type}): "
                f"ELO {agent.elo_rating:.0f}, "
                f"Domain ELO {domain_elo:.0f}, "
                f"Expertise {expertise:.0%}"
            )

        if len(all_scored) > len(team):
            lines.append("")
            lines.append(f"Considered {len(all_scored)} candidates, selected top {len(team)}")

        return "\n".join(lines)

    def update_from_result(
        self,
        team: TeamComposition,
        result: Any,
    ):
        """Update agent profiles based on debate result."""
        if not hasattr(result, "scores") or not result.scores:
            return

        # Update success rates and ELOs
        for agent in team.agents:
            if agent.name in result.scores:
                score = result.scores[agent.name]

                # Update success rate with exponential moving average
                alpha = 0.1
                success = 1.0 if score > 0.5 else 0.5 if score > 0.3 else 0.0
                agent.success_rate = alpha * success + (1 - alpha) * agent.success_rate

        # Record to history
        self._selection_history.append({
            "task_id": team.task_id,
            "selected": [a.name for a in team.agents],
            "result": "success" if getattr(result, "consensus_reached", False) else "no_consensus",
            "confidence": getattr(result, "confidence", 0),
            "timestamp": datetime.now().isoformat(),
        })

    def get_selection_history(self, limit: Optional[int] = None) -> list[dict]:
        """Retrieve selection history for meta-analysis."""
        history = self._selection_history.copy()
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        if limit:
            history = history[:limit]
        return history

    def get_best_team_combinations(self, min_debates: int = 3) -> list[dict]:
        """Analyze history to find best-performing team combinations."""
        from collections import defaultdict
        team_stats: dict[frozenset, dict] = defaultdict(lambda: {"wins": 0, "total": 0, "agents": []})

        for entry in self._selection_history:
            selected = entry.get('selected', [])
            if not selected:
                continue
            team_key = frozenset(selected)
            team_stats[team_key]["agents"] = sorted(selected)
            team_stats[team_key]["total"] += 1
            if entry.get('result') == 'success':
                team_stats[team_key]["wins"] += 1

        results = []
        for team_key, stats in team_stats.items():
            if stats["total"] >= min_debates:
                results.append({
                    "agents": stats["agents"],
                    "success_rate": stats["wins"] / stats["total"],
                    "total_debates": stats["total"],
                    "wins": stats["wins"],
                })
        results.sort(key=lambda x: x["success_rate"], reverse=True)
        return results

    def get_leaderboard(self, domain: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Get agent leaderboard."""
        agents = list(self.agent_pool.values())

        if domain:
            # Sort by domain-specific rating
            agents.sort(
                key=lambda a: a.domain_ratings.get(domain, a.elo_rating),
                reverse=True,
            )
        else:
            # Sort by overall score
            agents.sort(key=lambda a: a.overall_score, reverse=True)

        return [
            {
                "name": a.name,
                "type": a.agent_type,
                "elo": a.elo_rating,
                "domain_elo": a.domain_ratings.get(domain, a.elo_rating) if domain else None,
                "success_rate": a.success_rate,
                "overall_score": a.overall_score,
                "on_bench": a.name in self.bench,
            }
            for a in agents[:limit]
        ]

    def get_recommendations(
        self,
        requirements: TaskRequirements,
        limit: int = 5,
    ) -> list[dict]:
        """Get agent recommendations for a task."""
        candidates = list(self.agent_pool.values())
        scored = [(a, self._score_for_task(a, requirements)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "name": a.name,
                "type": a.agent_type,
                "match_score": score,
                "domain_expertise": a.expertise.get(requirements.primary_domain, 0),
                "reasoning": self._explain_match(a, requirements),
            }
            for a, score in scored[:limit]
        ]

    def _explain_match(self, agent: AgentProfile, requirements: TaskRequirements) -> str:
        """Explain why an agent matches requirements."""
        reasons = []

        exp = agent.expertise.get(requirements.primary_domain, 0)
        if exp > 0.7:
            reasons.append(f"Strong {requirements.primary_domain} expertise ({exp:.0%})")
        elif exp > 0.4:
            reasons.append(f"Moderate {requirements.primary_domain} expertise")

        if agent.elo_rating > 1600:
            reasons.append("High overall rating")

        matching_traits = [t for t in requirements.required_traits if t in agent.traits]
        if matching_traits:
            reasons.append(f"Has traits: {', '.join(matching_traits)}")

        if agent.success_rate > 0.8:
            reasons.append("Excellent success rate")

        return "; ".join(reasons) if reasons else "General purpose agent"

    def auto_route(
        self,
        task_text: str,
        task_id: Optional[str] = None,
        exclude: Optional[list[str]] = None,
    ) -> TeamComposition:
        """
        Automatically route a task to the best team based on detected domain.

        This is a convenience method that:
        1. Detects the domain from task text
        2. Creates TaskRequirements
        3. Selects the optimal team

        Args:
            task_text: Natural language task description
            task_id: Optional task identifier
            exclude: Agent names to exclude

        Returns:
            TeamComposition with selected agents

        Example:
            team = selector.auto_route("Add rate limiting to the API endpoints")
            # Returns team with agents strong in security and API domains
        """
        detector = DomainDetector()
        requirements = detector.get_task_requirements(task_text, task_id)

        # Detailed routing decision log
        logger.info(
            f"[ROUTING] Task '{task_id or 'unnamed'}': "
            f"primary_domain={requirements.primary_domain}, "
            f"secondary={requirements.secondary_domains}, "
            f"traits={requirements.required_traits}"
        )

        # Log detected domains with confidence
        domain_scores = detector.detect(task_text, top_n=5)
        if domain_scores:
            domain_breakdown = ", ".join(f"{d}:{c:.2f}" for d, c in domain_scores)
            logger.debug(f"[ROUTING] Domain scores: {domain_breakdown}")

        team = self.select_team(requirements, exclude=exclude)

        # Log team selection rationale
        agent_details = []
        for agent in team.agents:
            exp = agent.expertise.get(requirements.primary_domain, 0)
            agent_details.append(f"{agent.name}(exp={exp:.0%},elo={agent.elo_rating:.0f})")
        logger.info(
            f"[ROUTING] Selected team for {requirements.primary_domain}: "
            f"{', '.join(agent_details)}"
        )

        return team

    def get_domain_leaderboard(self, domain: str, limit: int = 10) -> list[dict]:
        """
        Get agent leaderboard for a specific domain.

        Args:
            domain: The domain to rank by (e.g., "security", "performance")
            limit: Maximum agents to return

        Returns:
            List of agent rankings with domain-specific scores
        """
        agents = list(self.agent_pool.values())

        # Score agents by domain expertise + domain ELO
        def domain_score(agent: AgentProfile) -> float:
            expertise = agent.expertise.get(domain, 0.5)
            domain_elo = agent.domain_ratings.get(domain, agent.elo_rating)
            # Weight: 40% expertise, 40% domain ELO, 20% overall
            return (
                expertise * 0.4 +
                (domain_elo - 1000) / 1000 * 0.4 +
                agent.overall_score * 0.2
            )

        agents.sort(key=domain_score, reverse=True)

        return [
            {
                "rank": i + 1,
                "name": a.name,
                "type": a.agent_type,
                "domain_score": domain_score(a),
                "expertise": a.expertise.get(domain, 0.5),
                "domain_elo": a.domain_ratings.get(domain, a.elo_rating),
                "overall_elo": a.elo_rating,
                "on_bench": a.name in self.bench,
            }
            for i, a in enumerate(agents[:limit])
        ]

    @classmethod
    def create_with_defaults(
        cls,
        elo_system: Optional[Any] = None,
        persona_manager: Optional[Any] = None,
    ) -> "AgentSelector":
        """
        Create an AgentSelector with default agent expertise profiles.

        This factory method initializes the selector with predefined
        expertise for common agents (Claude, Codex, Gemini, Grok, DeepSeek).

        Args:
            elo_system: Optional EloSystem for rating data
            persona_manager: Optional PersonaManager for dynamic traits

        Returns:
            Configured AgentSelector
        """
        selector = cls(elo_system=elo_system, persona_manager=persona_manager)

        # Register agents with default expertise
        for agent_name, expertise in DEFAULT_AGENT_EXPERTISE.items():
            profile = AgentProfile(
                name=agent_name,
                agent_type=agent_name,
                expertise=expertise.copy(),
            )
            selector.register_agent(profile)

        # Sync from ELO if available
        if elo_system:
            selector.refresh_from_elo_system()

        return selector
