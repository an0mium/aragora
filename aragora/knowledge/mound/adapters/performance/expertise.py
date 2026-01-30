"""
Expertise storage, retrieval, domain detection, and cache management.

Handles:
- Agent expertise storage with confidence tracking
- Domain expert queries with TTL caching
- Domain detection from debate questions
- Cache invalidation and statistics
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, cast

from aragora.knowledge.mound.adapters.performance.models import AgentExpertise

logger = logging.getLogger(__name__)


class _ExpertiseHostProtocol(Protocol):
    """Protocol for host class of ExpertiseMixin."""

    EXPERTISE_PREFIX: str
    MIN_ELO_CHANGE: int
    MIN_DEBATES_FOR_CONFIDENCE: int
    DOMAIN_KEYWORDS: dict[str, list[str]]
    _expertise: dict[str, dict[str, Any]]
    _agent_history: dict[str, list[dict[str, Any]]]
    _domain_agents: dict[str, list[str]]
    _agent_domains: dict[str, list[str]]
    _domain_experts_cache: dict[str, tuple[float, Any]]
    _cache_hits: int
    _cache_misses: int
    _cache_ttl_seconds: float


class ExpertiseMixin(_ExpertiseHostProtocol):
    """Mixin providing expertise storage, retrieval, and caching methods.

    Expects the following attributes on the host class:
    - EXPERTISE_PREFIX: str
    - MIN_ELO_CHANGE: int
    - MIN_DEBATES_FOR_CONFIDENCE: int
    - DOMAIN_KEYWORDS: dict[str, list[str]]
    - _expertise: dict[str, dict[str, Any]]
    - _agent_history: dict[str, list[dict[str, Any]]]
    - _domain_agents: dict[str, list[str]]
    - _agent_domains: dict[str, list[str]]
    - _domain_experts_cache: dict[str, tuple]
    - _cache_hits: int
    - _cache_misses: int
    - _cache_ttl_seconds: float
    """

    # =========================================================================
    # Expertise Storage Methods
    # =========================================================================

    def store_agent_expertise(
        self,
        agent_name: str,
        domain: str,
        elo: float,
        delta: float = 0.0,
        debate_id: str | None = None,
    ) -> str | None:
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

        # Invalidate cache for this domain
        self.invalidate_domain_cache(domain)

        logger.info(
            f"Stored expertise: {agent_name} in {domain} -> {elo} (confidence={confidence:.2f})"
        )
        return expertise_id

    # =========================================================================
    # Expertise Retrieval Methods
    # =========================================================================

    def get_agent_expertise(
        self,
        agent_name: str,
        domain: str | None = None,
    ) -> Optional[dict[str, Any]]:
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
    ) -> list[AgentExpertise]:
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
        # Check cache first
        cache_key = f"{domain}:{limit}:{min_confidence}"
        if use_cache and cache_key in self._domain_experts_cache:
            timestamp, cached_results = self._domain_experts_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                self._cache_hits += 1
                logger.debug(f"Cache hit for domain experts: {domain}")
                return cast(list[AgentExpertise], cached_results)
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

    def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
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

    # =========================================================================
    # Domain Detection
    # =========================================================================

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

    def get_all_domains(self) -> list[str]:
        """Get all domains with stored expertise."""
        return list(self._domain_agents.keys())

    # =========================================================================
    # Cache Management
    # =========================================================================

    def invalidate_domain_cache(self, domain: str | None = None) -> int:
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

    def get_cache_stats(self) -> dict[str, Any]:
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


__all__ = ["ExpertiseMixin"]
