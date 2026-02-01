"""
Knowledge Mound persistence methods for the PerformanceAdapter.

Handles:
- Syncing expertise data to the Knowledge Mound
- Loading expertise data from the Knowledge Mound
- Converting rating dicts to KnowledgeItem format
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.unified.types import KnowledgeItem

logger = logging.getLogger(__name__)


class _KMPersistenceHostProtocol(Protocol):
    """Protocol for host class of KMPersistenceMixin."""

    EXPERTISE_PREFIX: str
    _expertise: dict[str, dict[str, Any]]
    _domain_agents: dict[str, list[str]]
    _agent_domains: dict[str, list[str]]


class KMPersistenceMixin:
    """Mixin providing Knowledge Mound sync/load and conversion methods.

    NOTE: Does NOT inherit from Protocol to preserve cooperative inheritance.

    Expects the following attributes on the host class:
    - EXPERTISE_PREFIX: str
    - _expertise: dict[str, dict[str, Any]]
    - _domain_agents: dict[str, list[str]]
    - _agent_domains: dict[str, list[str]]
    """

    # =========================================================================
    # Knowledge Item Conversion
    # =========================================================================

    def to_knowledge_item(self, rating: dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a rating dict to a KnowledgeItem.

        Args:
            rating: The rating dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Confidence based on games played (more games = more confident rating)
        games = rating.get("games_played", 0)
        if games >= 50:
            confidence = ConfidenceLevel.VERIFIED
        elif games >= 20:
            confidence = ConfidenceLevel.HIGH
        elif games >= 10:
            confidence = ConfidenceLevel.MEDIUM
        elif games >= 3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = rating.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        elo = rating.get("elo", 1000)
        # Normalize ELO to 0-1 importance (1000 = 0.5, 1500 = 0.75, 2000 = 1.0)
        importance = min(1.0, max(0.0, (elo - 500) / 1500))

        content = (
            f"Agent {rating.get('agent_name', 'unknown')}: "
            f"ELO {elo:.0f}, "
            f"{rating.get('wins', 0)}W/{rating.get('losses', 0)}L/{rating.get('draws', 0)}D"
        )

        return KnowledgeItem(
            id=rating["id"],
            content=content,
            source=KnowledgeSource.ELO,
            source_id=rating.get("agent_name", rating["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "agent_name": rating.get("agent_name", ""),
                "elo": elo,
                "win_rate": rating.get("win_rate", 0.0),
                "games_played": games,
                "domain_elos": rating.get("domain_elos", {}),
                "calibration_accuracy": rating.get("calibration_accuracy", 0.0),
                "reason": rating.get("reason", ""),
            },
            importance=importance,
        )

    # =========================================================================
    # KM Persistence Methods
    # =========================================================================

    async def sync_to_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> dict[str, Any]:
        """
        Persist all expertise data to the Knowledge Mound.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID for storage

        Returns:
            Dict with sync statistics
        """
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        result: dict[str, Any] = {
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
            f"Performance sync to KM: expertise={result['expertise_synced']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    async def load_from_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> dict[str, Any]:
        """
        Load expertise data from the Knowledge Mound.

        This restores adapter state from KM persistence.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID to load from

        Returns:
            Dict with load statistics
        """
        result: dict[str, Any] = {
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
            f"Performance load from KM: loaded={result['expertise_loaded']}, "
            f"errors={len(result['errors'])}"
        )
        return result


__all__ = ["KMPersistenceMixin"]
