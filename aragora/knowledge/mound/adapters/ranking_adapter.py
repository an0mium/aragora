"""
RankingAdapter - Bridges ELO/Ranking system to the Knowledge Mound.

This adapter enables bidirectional integration between the ranking system
and the Knowledge Mound:

- Data flow IN: Agent expertise profiles stored in KM
- Data flow OUT: Domain experts retrieved for team selection
- Reverse flow: KM performance history informs ELO initialization

The adapter provides:
- Agent expertise storage after ELO changes
- Domain expert retrieval for team selection
- Historical performance trends
- Cross-debate expertise tracking

ID Prefixes:
- ex_: Expertise records
- dm_: Domain mappings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ExpertiseSearchResult:
    """Wrapper for expertise search results with adapter metadata."""

    expertise: Dict[str, Any]
    relevance_score: float = 0.0
    matched_domains: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_domains is None:
            self.matched_domains = []


@dataclass
class AgentExpertise:
    """Represents an agent's expertise in a domain."""

    agent_name: str
    domain: str
    elo: float
    confidence: float  # Based on number of debates
    last_updated: str
    debate_count: int = 0


class RankingAdapter:
    """
    Adapter that bridges EloSystem to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - store_agent_expertise: Store agent domain expertise after ELO change
    - get_domain_experts: Retrieve top agents for a domain
    - get_agent_history: Get historical performance for an agent
    - detect_domain: Detect domain from debate question

    Usage:
        from aragora.ranking.elo import EloSystem
        from aragora.knowledge.mound.adapters import RankingAdapter

        elo = EloSystem()
        adapter = RankingAdapter(elo)

        # After ELO update, store expertise
        adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1650,
            delta=50,
            debate_id="debate-123",
        )

        # For team selection, query domain experts
        experts = adapter.get_domain_experts("security", limit=5)
    """

    EXPERTISE_PREFIX = "ex_"
    DOMAIN_PREFIX = "dm_"

    # Thresholds
    MIN_ELO_CHANGE = 25  # Minimum ELO change to record
    MIN_DEBATES_FOR_CONFIDENCE = 5  # Debates needed for high confidence

    # Domain keywords for detection (order matters - first match wins)
    # Security terms are checked first to catch security-related SQL questions
    DOMAIN_KEYWORDS = {
        "security": ["security", "vulnerability", "exploit", "attack", "defense", "crypto", "auth", "injection", "xss", "csrf"],
        "coding": ["code", "programming", "implementation", "algorithm", "function", "class"],
        "architecture": ["architecture", "design", "pattern", "system", "scalable", "microservice"],
        "testing": ["test", "qa", "quality", "bug", "regression", "coverage"],
        "data": ["data", "database", "sql", "analytics", "ml", "machine learning", "ai"],
        "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure", "cloud"],
        "legal": ["legal", "compliance", "regulation", "contract", "liability", "gdpr"],
        "ethics": ["ethics", "moral", "fair", "bias", "responsible", "privacy"],
    }

    def __init__(
        self,
        elo_system: Optional[Any] = None,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            elo_system: Optional EloSystem instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._elo_system = elo_system
        self._enable_dual_write = enable_dual_write

        # In-memory storage for queries (will be replaced by KM backend)
        self._expertise: Dict[str, Dict[str, Any]] = {}  # {agent_domain: expertise_data}
        self._agent_history: Dict[str, List[Dict[str, Any]]] = {}  # {agent_name: [records]}

        # Indices for fast lookup
        self._domain_agents: Dict[str, List[str]] = {}  # {domain: [agent_names]}
        self._agent_domains: Dict[str, List[str]] = {}  # {agent_name: [domains]}

    @property
    def elo_system(self) -> Optional[Any]:
        """Access the underlying EloSystem."""
        return self._elo_system

    def set_elo_system(self, elo_system: Any) -> None:
        """Set the ELO system to use."""
        self._elo_system = elo_system

    def store_agent_expertise(
        self,
        agent_name: str,
        domain: str,
        elo: float,
        delta: float = 0.0,
        debate_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store agent expertise in the Knowledge Mound.

        Args:
            agent_name: Name of the agent
            domain: Domain of expertise
            elo: Current ELO rating
            delta: ELO change from last update
            debate_id: Optional debate ID that triggered update

        Returns:
            The expertise ID if stored, None if below threshold
        """
        # Skip minor changes
        if abs(delta) < self.MIN_ELO_CHANGE:
            logger.debug(f"ELO change too small for {agent_name}: {delta}")
            return None

        expertise_key = f"{agent_name}:{domain}"
        expertise_id = f"{self.EXPERTISE_PREFIX}{expertise_key.replace(':', '_')}"

        # Get or create expertise record
        existing = self._expertise.get(expertise_key)
        debate_count = 1

        if existing:
            debate_count = existing.get("debate_count", 0) + 1

        # Calculate confidence based on debate count
        confidence = min(1.0, debate_count / self.MIN_DEBATES_FOR_CONFIDENCE)

        expertise_data = {
            "id": expertise_id,
            "agent_name": agent_name,
            "domain": domain,
            "elo": elo,
            "previous_elo": existing.get("elo") if existing else elo - delta,
            "delta": delta,
            "confidence": confidence,
            "debate_count": debate_count,
            "last_debate_id": debate_id,
            "created_at": existing.get("created_at") if existing else datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        self._expertise[expertise_key] = expertise_data

        # Update indices
        if domain not in self._domain_agents:
            self._domain_agents[domain] = []
        if agent_name not in self._domain_agents[domain]:
            self._domain_agents[domain].append(agent_name)

        if agent_name not in self._agent_domains:
            self._agent_domains[agent_name] = []
        if domain not in self._agent_domains[agent_name]:
            self._agent_domains[agent_name].append(domain)

        # Store in history
        if agent_name not in self._agent_history:
            self._agent_history[agent_name] = []
        self._agent_history[agent_name].append({
            "domain": domain,
            "elo": elo,
            "delta": delta,
            "debate_id": debate_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(f"Stored expertise: {agent_name} in {domain} -> {elo} (confidence={confidence:.2f})")
        return expertise_id

    def get_agent_expertise(
        self,
        agent_name: str,
        domain: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get expertise record for an agent.

        Args:
            agent_name: Agent name
            domain: Optional specific domain (returns all if not specified)

        Returns:
            Expertise dict or None
        """
        if domain:
            expertise_key = f"{agent_name}:{domain}"
            return self._expertise.get(expertise_key)

        # Return all domains for this agent
        domains = self._agent_domains.get(agent_name, [])
        result = {}
        for d in domains:
            expertise_key = f"{agent_name}:{d}"
            if expertise_key in self._expertise:
                result[d] = self._expertise[expertise_key]
        return result if result else None

    def get_domain_experts(
        self,
        domain: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[AgentExpertise]:
        """
        Get top experts for a domain.

        Args:
            domain: Domain to query
            limit: Maximum experts to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of AgentExpertise sorted by ELO descending
        """
        agents = self._domain_agents.get(domain, [])
        results = []

        for agent_name in agents:
            expertise_key = f"{agent_name}:{domain}"
            expertise = self._expertise.get(expertise_key)

            if expertise and expertise.get("confidence", 0) >= min_confidence:
                results.append(AgentExpertise(
                    agent_name=agent_name,
                    domain=domain,
                    elo=expertise.get("elo", 1500),
                    confidence=expertise.get("confidence", 0.0),
                    last_updated=expertise.get("updated_at", ""),
                    debate_count=expertise.get("debate_count", 0),
                ))

        # Sort by ELO descending
        results.sort(key=lambda x: x.elo, reverse=True)

        return results[:limit]

    def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical performance records for an agent.

        Args:
            agent_name: Agent name
            limit: Maximum records to return
            domain: Optional domain filter

        Returns:
            List of history records (newest first)
        """
        history = self._agent_history.get(agent_name, [])

        if domain:
            history = [h for h in history if h.get("domain") == domain]

        # Sort by timestamp descending
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return history[:limit]

    def detect_domain(self, question: str) -> str:
        """
        Detect domain from debate question.

        Args:
            question: The debate question

        Returns:
            Detected domain name (defaults to "general")
        """
        question_lower = question.lower()

        # Check each domain's keywords
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    logger.debug(f"Detected domain '{domain}' from keyword '{keyword}'")
                    return domain

        return "general"

    def get_all_domains(self) -> List[str]:
        """Get all domains with stored expertise."""
        return list(self._domain_agents.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored expertise."""
        return {
            "total_expertise_records": len(self._expertise),
            "total_agents": len(self._agent_domains),
            "total_domains": len(self._domain_agents),
            "agents_per_domain": {
                domain: len(agents)
                for domain, agents in self._domain_agents.items()
            },
            "total_history_records": sum(len(h) for h in self._agent_history.values()),
        }


__all__ = [
    "RankingAdapter",
    "AgentExpertise",
    "ExpertiseSearchResult",
]
