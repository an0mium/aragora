"""
Knowledge Query Skill.

Provides integration with Aragora's Knowledge Mound for querying
accumulated knowledge, consensus positions, and debate outcomes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


class KnowledgeQuerySkill(Skill):
    """
    Skill for querying the Knowledge Mound.

    Provides access to:
    - Consensus positions from past debates
    - Evidence and citations
    - Learned patterns and insights
    - Cross-debate knowledge
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="knowledge_query",
            version="1.0.0",
            description="Query the Knowledge Mound for accumulated knowledge",
            capabilities=[
                SkillCapability.KNOWLEDGE_QUERY,
                SkillCapability.READ_DATABASE,
            ],
            input_schema={
                "query": {
                    "type": "string",
                    "description": "Natural language query or topic",
                    "required": True,
                },
                "query_type": {
                    "type": "string",
                    "description": "Type of query: semantic, keyword, pattern",
                    "default": "semantic",
                },
                "sources": {
                    "type": "array",
                    "description": "Knowledge sources to query (consensus, evidence, patterns)",
                    "default": ["consensus", "evidence"],
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum results to return",
                    "default": 10,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence score (0-1)",
                    "default": 0.5,
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID for multi-tenant isolation",
                },
            },
            tags=["knowledge", "query", "search"],
            debate_compatible=True,
            requires_debate_context=False,
            max_execution_time_seconds=30.0,
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute knowledge query."""
        query = input_data.get("query", "")
        if not query:
            return SkillResult.create_failure(
                "Query is required",
                error_code="missing_query",
            )

        query_type = input_data.get("query_type", "semantic")
        sources = input_data.get("sources", ["consensus", "evidence"])
        max_results = input_data.get("max_results", 10)
        min_confidence = input_data.get("min_confidence", 0.5)
        tenant_id = input_data.get("tenant_id") or context.tenant_id

        try:
            # Try to get Knowledge Mound
            mound = await self._get_knowledge_mound()
            if not mound:
                return SkillResult.create_failure(
                    "Knowledge Mound not available",
                    error_code="service_unavailable",
                )

            results: Dict[str, Any] = {
                "query": query,
                "query_type": query_type,
                "sources_queried": sources,
            }

            # Query each requested source
            if "consensus" in sources:
                consensus_results = await self._query_consensus(
                    mound, query, max_results, min_confidence, tenant_id
                )
                results["consensus"] = consensus_results

            if "evidence" in sources:
                evidence_results = await self._query_evidence(
                    mound, query, max_results, min_confidence, tenant_id
                )
                results["evidence"] = evidence_results

            if "patterns" in sources:
                pattern_results = await self._query_patterns(mound, query, max_results, tenant_id)
                results["patterns"] = pattern_results

            if "insights" in sources:
                insight_results = await self._query_insights(mound, query, max_results, tenant_id)
                results["insights"] = insight_results

            # Count total results
            total = sum(len(results.get(source, [])) for source in sources)
            results["total_results"] = total

            return SkillResult.create_success(results)

        except Exception as e:
            logger.exception(f"Knowledge query failed: {e}")
            return SkillResult.create_failure(f"Query failed: {e}")

    async def _get_knowledge_mound(self) -> Optional[Any]:
        """Get the Knowledge Mound instance."""
        try:
            from aragora.knowledge.mound import get_knowledge_mound

            return await get_knowledge_mound()
        except ImportError:
            logger.debug("Knowledge Mound not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to get Knowledge Mound: {e}")
            return None

    async def _query_consensus(
        self,
        mound: Any,
        query: str,
        max_results: int,
        min_confidence: float,
        tenant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Query consensus positions."""
        try:
            if hasattr(mound, "query_consensus"):
                results = await mound.query_consensus(
                    query=query,
                    limit=max_results,
                    min_confidence=min_confidence,
                    tenant_id=tenant_id,
                )
                return [
                    {
                        "topic": r.topic if hasattr(r, "topic") else str(r),
                        "position": r.position if hasattr(r, "position") else None,
                        "confidence": r.confidence if hasattr(r, "confidence") else 0.0,
                        "debate_id": r.debate_id if hasattr(r, "debate_id") else None,
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Consensus query error: {e}")
        return []

    async def _query_evidence(
        self,
        mound: Any,
        query: str,
        max_results: int,
        min_confidence: float,
        tenant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Query evidence records."""
        try:
            if hasattr(mound, "query_evidence"):
                results = await mound.query_evidence(
                    query=query,
                    limit=max_results,
                    min_relevance=min_confidence,
                    tenant_id=tenant_id,
                )
                return [
                    {
                        "claim": r.claim if hasattr(r, "claim") else str(r),
                        "source": r.source if hasattr(r, "source") else None,
                        "relevance": r.relevance if hasattr(r, "relevance") else 0.0,
                        "url": r.url if hasattr(r, "url") else None,
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Evidence query error: {e}")
        return []

    async def _query_patterns(
        self,
        mound: Any,
        query: str,
        max_results: int,
        tenant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Query learned patterns."""
        try:
            if hasattr(mound, "query_patterns"):
                results = await mound.query_patterns(
                    query=query,
                    limit=max_results,
                    tenant_id=tenant_id,
                )
                return [
                    {
                        "pattern": r.pattern if hasattr(r, "pattern") else str(r),
                        "frequency": r.frequency if hasattr(r, "frequency") else 0,
                        "context": r.context if hasattr(r, "context") else None,
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Pattern query error: {e}")
        return []

    async def _query_insights(
        self,
        mound: Any,
        query: str,
        max_results: int,
        tenant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Query meta-learning insights."""
        try:
            if hasattr(mound, "query_insights"):
                results = await mound.query_insights(
                    query=query,
                    limit=max_results,
                    tenant_id=tenant_id,
                )
                return [
                    {
                        "insight": r.insight if hasattr(r, "insight") else str(r),
                        "category": r.category if hasattr(r, "category") else None,
                        "confidence": r.confidence if hasattr(r, "confidence") else 0.0,
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Insight query error: {e}")
        return []


# Skill instance for registration
SKILLS = [KnowledgeQuerySkill()]
