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
from datetime import datetime, timezone
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
        "security": [
            "security",
            "vulnerability",
            "exploit",
            "attack",
            "defense",
            "crypto",
            "auth",
            "injection",
            "xss",
            "csrf",
        ],
        "coding": ["code", "programming", "implementation", "algorithm", "function", "class"],
        "architecture": ["architecture", "design", "pattern", "system", "scalable", "microservice"],
        "testing": ["test", "qa", "quality", "bug", "regression", "coverage"],
        "data": ["data", "database", "sql", "analytics", "ml", "machine learning", "ai"],
        "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure", "cloud"],
        "legal": ["legal", "compliance", "regulation", "contract", "liability", "gdpr"],
        "ethics": ["ethics", "moral", "fair", "bias", "responsible", "privacy"],
    }

    # Cache configuration
    DEFAULT_CACHE_TTL_SECONDS = 60.0  # Default TTL for cached queries

    def __init__(
        self,
        elo_system: Optional[Any] = None,
        enable_dual_write: bool = False,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ):
        """
        Initialize the adapter.

        Args:
            elo_system: Optional EloSystem instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
            cache_ttl_seconds: TTL for cached queries (default: 60 seconds)
        """
        self._elo_system = elo_system
        self._enable_dual_write = enable_dual_write
        self._cache_ttl_seconds = cache_ttl_seconds

        # In-memory storage for queries (will be replaced by KM backend)
        self._expertise: Dict[str, Dict[str, Any]] = {}  # {agent_domain: expertise_data}
        self._agent_history: Dict[str, List[Dict[str, Any]]] = {}  # {agent_name: [records]}

        # Query cache with TTL
        self._domain_experts_cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, results)}
        self._cache_hits = 0
        self._cache_misses = 0

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
            "created_at": (
                existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat()
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
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
        self._agent_history[agent_name].append(
            {
                "domain": domain,
                "elo": elo,
                "delta": delta,
                "debate_id": debate_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Invalidate cache for this domain to ensure fresh data
        self.invalidate_domain_cache(domain)

        logger.info(
            f"Stored expertise: {agent_name} in {domain} -> {elo} (confidence={confidence:.2f})"
        )
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
        use_cache: bool = True,
    ) -> List[AgentExpertise]:
        """
        Get top experts for a domain.

        Args:
            domain: Domain to query
            limit: Maximum experts to return
            min_confidence: Minimum confidence threshold
            use_cache: Whether to use cached results (default: True)

        Returns:
            List of AgentExpertise sorted by ELO descending
        """
        import time

        # Check cache first
        cache_key = f"{domain}:{limit}:{min_confidence}"
        if use_cache and cache_key in self._domain_experts_cache:
            timestamp, cached_results = self._domain_experts_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                self._cache_hits += 1
                logger.debug(f"Cache hit for domain experts: {domain}")
                return cached_results
            else:
                # Cache expired, remove it
                del self._domain_experts_cache[cache_key]

        self._cache_misses += 1

        # Query fresh data
        agents = self._domain_agents.get(domain, [])
        results = []

        for agent_name in agents:
            expertise_key = f"{agent_name}:{domain}"
            expertise = self._expertise.get(expertise_key)

            if expertise and expertise.get("confidence", 0) >= min_confidence:
                results.append(
                    AgentExpertise(
                        agent_name=agent_name,
                        domain=domain,
                        elo=expertise.get("elo", 1500),
                        confidence=expertise.get("confidence", 0.0),
                        last_updated=expertise.get("updated_at", ""),
                        debate_count=expertise.get("debate_count", 0),
                    )
                )

        # Sort by ELO descending
        results.sort(key=lambda x: x.elo, reverse=True)
        results = results[:limit]

        # Cache results
        if use_cache:
            self._domain_experts_cache[cache_key] = (time.time(), results)
            logger.debug(f"Cached domain experts for {domain}: {len(results)} results")

        return results

    def invalidate_domain_cache(self, domain: Optional[str] = None) -> int:
        """
        Invalidate the domain experts cache.

        Args:
            domain: If specified, only invalidate cache for this domain.
                   If None, invalidate all cached entries.

        Returns:
            Number of cache entries invalidated
        """
        if domain is None:
            count = len(self._domain_experts_cache)
            self._domain_experts_cache.clear()
            return count

        # Invalidate entries for specific domain
        keys_to_remove = [k for k in self._domain_experts_cache if k.startswith(f"{domain}:")]
        for key in keys_to_remove:
            del self._domain_experts_cache[key]
        return len(keys_to_remove)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache_hits, cache_misses, cache_size, hit_rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._domain_experts_cache),
            "hit_rate": hit_rate,
            "ttl_seconds": self._cache_ttl_seconds,
        }

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
                domain: len(agents) for domain, agents in self._domain_agents.items()
            },
            "total_history_records": sum(len(h) for h in self._agent_history.values()),
        }

    # =========================================================================
    # Knowledge Mound Persistence Methods
    # =========================================================================

    async def sync_to_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Persist all expertise data to the Knowledge Mound.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID for storage

        Returns:
            Dict with sync statistics
        """
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        result: Dict[str, Any] = {
            "expertise_synced": 0,
            "history_synced": 0,
            "errors": [],
        }

        # Sync expertise records
        for expertise_key, expertise_data in self._expertise.items():
            try:
                content = (
                    f"Agent: {expertise_data['agent_name']}\n"
                    f"Domain: {expertise_data['domain']}\n"
                    f"ELO: {expertise_data['elo']}\n"
                    f"Confidence: {expertise_data['confidence']:.2f}\n"
                    f"Debates: {expertise_data['debate_count']}"
                )

                request = IngestionRequest(
                    content=content,
                    source_type=SourceType.RANKING,
                    workspace_id=workspace_id,
                    confidence=expertise_data["confidence"],
                    tier="slow",  # Slow tier for expertise data
                    metadata={
                        "type": "agent_expertise",
                        "expertise_id": expertise_data["id"],
                        "agent_name": expertise_data["agent_name"],
                        "domain": expertise_data["domain"],
                        "elo": expertise_data["elo"],
                        "debate_count": expertise_data["debate_count"],
                    },
                )

                await mound.ingest(request)
                result["expertise_synced"] += 1

            except Exception as e:
                result["errors"].append(f"Expertise {expertise_key}: {e}")

        logger.info(
            f"Ranking sync to KM: expertise={result['expertise_synced']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    async def load_from_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Load expertise data from the Knowledge Mound.

        This restores adapter state from KM persistence.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID to load from

        Returns:
            Dict with load statistics
        """
        result: Dict[str, Any] = {
            "expertise_loaded": 0,
            "errors": [],
        }

        try:
            # Query KM for expertise records
            nodes = await mound.query_nodes(
                workspace_id=workspace_id,
                source_type="ranking",
                limit=1000,
            )

            for node in nodes:
                metadata = node.metadata or {}
                if metadata.get("type") != "agent_expertise":
                    continue

                agent_name = metadata.get("agent_name")
                domain = metadata.get("domain")
                if not agent_name or not domain:
                    continue

                expertise_key = f"{agent_name}:{domain}"
                expertise_id = f"{self.EXPERTISE_PREFIX}{expertise_key.replace(':', '_')}"

                self._expertise[expertise_key] = {
                    "id": expertise_id,
                    "agent_name": agent_name,
                    "domain": domain,
                    "elo": metadata.get("elo", 1500),
                    "confidence": (
                        metadata.get("confidence", node.confidence)
                        if hasattr(node, "confidence")
                        else 0.5
                    ),
                    "debate_count": metadata.get("debate_count", 0),
                    "created_at": (
                        node.created_at.isoformat()
                        if node.created_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "updated_at": (
                        node.updated_at.isoformat()
                        if node.updated_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                }

                # Update indices
                if domain not in self._domain_agents:
                    self._domain_agents[domain] = []
                if agent_name not in self._domain_agents[domain]:
                    self._domain_agents[domain].append(agent_name)

                if agent_name not in self._agent_domains:
                    self._agent_domains[agent_name] = []
                if domain not in self._agent_domains[agent_name]:
                    self._agent_domains[agent_name].append(domain)

                result["expertise_loaded"] += 1

        except Exception as e:
            result["errors"].append(f"Load failed: {e}")
            logger.error(f"Failed to load expertise from KM: {e}")

        logger.info(
            f"Ranking load from KM: loaded={result['expertise_loaded']}, "
            f"errors={len(result['errors'])}"
        )
        return result


__all__ = [
    "RankingAdapter",
    "AgentExpertise",
    "ExpertiseSearchResult",
]
