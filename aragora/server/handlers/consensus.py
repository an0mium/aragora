"""
Consensus Memory endpoint handlers.

Endpoints:
- GET /api/consensus/similar - Find debates similar to a topic
- GET /api/consensus/settled - Get high-confidence settled topics
- GET /api/consensus/stats - Get consensus memory statistics
- GET /api/consensus/dissents - Get recent dissenting views
- GET /api/consensus/contrarian-views - Get contrarian perspectives
- GET /api/consensus/risk-warnings - Get risk warnings and edge cases
- GET /api/consensus/domain/:domain - Get domain-specific history
"""

import json
import logging
from typing import Optional

from aragora.config import (
    CACHE_TTL_CONSENSUS_SIMILAR,
    CACHE_TTL_CONSENSUS_SETTLED,
    CACHE_TTL_CONSENSUS_STATS,
    CACHE_TTL_RECENT_DISSENTS,
    CACHE_TTL_CONTRARIAN_VIEWS,
    CACHE_TTL_RISK_WARNINGS,
)
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
    ttl_cache,
    require_feature,
    get_db_connection,
    handle_errors,
)
from aragora.utils.optional_imports import try_import

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies using centralized utility
_consensus_imports, CONSENSUS_MEMORY_AVAILABLE = try_import(
    "aragora.memory.consensus", "ConsensusMemory", "DissentRetriever"
)
ConsensusMemory = _consensus_imports["ConsensusMemory"]
DissentRetriever = _consensus_imports["DissentRetriever"]



class ConsensusHandler(BaseHandler):
    """Handler for consensus memory endpoints."""

    ROUTES = [
        "/api/consensus/similar",
        "/api/consensus/settled",
        "/api/consensus/stats",
        "/api/consensus/dissents",
        "/api/consensus/contrarian-views",
        "/api/consensus/risk-warnings",
        "/api/consensus/seed-demo",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/consensus/domain/:domain pattern
        if path.startswith("/api/consensus/domain/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route consensus requests to appropriate methods."""
        if path == "/api/consensus/similar":
            # Validate raw topic length before truncation
            raw_topic = query_params.get('topic', '')
            if isinstance(raw_topic, list):
                raw_topic = raw_topic[0] if raw_topic else ''
            if len(raw_topic) > 500:
                return error_response("Topic too long (max 500 chars)", 400)
            topic = get_bounded_string_param(query_params, 'topic', '', max_length=500)
            if not topic:
                return error_response("Topic required (max 500 chars)", 400)
            limit = get_clamped_int_param(query_params, 'limit', 5, min_val=1, max_val=20)
            return self._get_similar_debates(topic.strip(), limit)

        if path == "/api/consensus/settled":
            min_confidence = get_bounded_float_param(query_params, 'min_confidence', 0.8, min_val=0.0, max_val=1.0)
            limit = get_clamped_int_param(query_params, 'limit', 20, min_val=1, max_val=100)
            return self._get_settled_topics(min_confidence, limit)

        if path == "/api/consensus/stats":
            return self._get_consensus_stats()

        if path == "/api/consensus/dissents":
            topic = get_bounded_string_param(query_params, 'topic', '', max_length=500)
            domain = get_bounded_string_param(query_params, 'domain', None, max_length=100)
            limit = get_clamped_int_param(query_params, 'limit', 10, min_val=1, max_val=50)
            return self._get_recent_dissents(
                topic.strip() if topic else None,
                domain,
                limit
            )

        if path == "/api/consensus/contrarian-views":
            topic = get_bounded_string_param(query_params, 'topic', '', max_length=500)
            domain = get_bounded_string_param(query_params, 'domain', None, max_length=100)
            limit = get_clamped_int_param(query_params, 'limit', 10, min_val=1, max_val=50)
            return self._get_contrarian_views(
                topic.strip() if topic else None,
                domain,
                limit
            )

        if path == "/api/consensus/risk-warnings":
            topic = get_bounded_string_param(query_params, 'topic', '', max_length=500)
            domain = get_bounded_string_param(query_params, 'domain', None, max_length=100)
            limit = get_clamped_int_param(query_params, 'limit', 10, min_val=1, max_val=50)
            return self._get_risk_warnings(
                topic.strip() if topic else None,
                domain,
                limit
            )

        if path == "/api/consensus/seed-demo":
            return self._seed_demo_data()

        if path.startswith("/api/consensus/domain/"):
            domain, err = self.extract_path_param(path, 3, "domain")
            if err:
                return err
            limit = get_clamped_int_param(query_params, 'limit', 50, min_val=1, max_val=200)
            return self._get_domain_history(domain, limit)

        return None

    @ttl_cache(ttl_seconds=CACHE_TTL_CONSENSUS_SIMILAR, key_prefix="consensus_similar", skip_first=True)
    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("similar debates retrieval")
    def _get_similar_debates(self, topic: str, limit: int) -> HandlerResult:
        """Find debates similar to a topic."""
        if not topic:
            return error_response("topic parameter required", 400)

        memory = ConsensusMemory()
        similar = memory.find_similar_debates(topic, limit=limit)
        return json_response({
            "query": topic,
            "similar": [
                {
                    "topic": s.consensus.topic,
                    "conclusion": s.consensus.conclusion,
                    "strength": s.consensus.strength.value,
                    "confidence": s.consensus.confidence,
                    "similarity": s.similarity_score,
                    "agents": s.consensus.participating_agents,
                    "dissent_count": len(s.dissents),
                    "timestamp": s.consensus.timestamp.isoformat(),
                }
                for s in similar
            ],
            "count": len(similar),
        })

    @ttl_cache(ttl_seconds=CACHE_TTL_CONSENSUS_SETTLED, key_prefix="consensus_settled", skip_first=True)
    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("settled topics retrieval")
    def _get_settled_topics(self, min_confidence: float, limit: int) -> HandlerResult:
        """Get high-confidence settled topics."""
        memory = ConsensusMemory()
        with get_db_connection(memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT topic, conclusion, confidence, strength, timestamp
                FROM consensus
                WHERE confidence >= ?
                ORDER BY confidence DESC, timestamp DESC
                LIMIT ?
            """, (min_confidence, limit))
            rows = cursor.fetchall()

        return json_response({
            "min_confidence": min_confidence,
            "topics": [
                {
                    "topic": row[0],
                    "conclusion": row[1],
                    "confidence": row[2],
                    "strength": row[3],
                    "timestamp": row[4],
                }
                for row in rows
            ],
            "count": len(rows),
        })

    @ttl_cache(ttl_seconds=CACHE_TTL_CONSENSUS_STATS, key_prefix="consensus_stats", skip_first=True)
    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("consensus stats retrieval")
    def _get_consensus_stats(self) -> HandlerResult:
        """Get consensus memory statistics."""
        memory = ConsensusMemory()
        raw_stats = memory.get_statistics()

        with get_db_connection(memory.db_path) as conn:
            cursor = conn.cursor()
            # Combined query for better performance
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN confidence >= 0.7 THEN 1 ELSE 0 END) as high_conf_count,
                    AVG(confidence) as avg_conf
                FROM consensus
            """)
            row = cursor.fetchone()
            high_confidence_count = row[0] if row and row[0] else 0
            avg_confidence = row[1] if row and row[1] else 0.0

        return json_response({
            "total_topics": raw_stats.get("total_consensus", 0),
            "high_confidence_count": high_confidence_count,
            "domains": list(raw_stats.get("by_domain", {}).keys()),
            "avg_confidence": round(avg_confidence, 3),
            "total_dissents": raw_stats.get("total_dissents", 0),
            "by_strength": raw_stats.get("by_strength", {}),
            "by_domain": raw_stats.get("by_domain", {}),
        })

    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("demo data seeding")
    def _seed_demo_data(self) -> HandlerResult:
        """Seed demo consensus data for search functionality."""
        try:
            from aragora.fixtures import load_demo_consensus
            memory = ConsensusMemory()
            seeded = load_demo_consensus(memory)
            return json_response({
                "success": True,
                "seeded": seeded,
                "message": f"Seeded {seeded} demo consensus records" if seeded > 0 else "Database already has data, skipped seeding",
            })
        except ImportError as e:
            return error_response(f"Fixtures module not available: {e}", 500)
        except Exception as e:
            logger.error(f"Failed to seed demo data: {e}")
            return error_response(f"Seeding failed: {e}", 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_RECENT_DISSENTS, key_prefix="recent_dissents", skip_first=True)
    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("recent dissents retrieval")
    def _get_recent_dissents(
        self, topic: Optional[str], domain: Optional[str], limit: int
    ) -> HandlerResult:
        """Get recent dissents, optionally filtered by topic."""
        memory = ConsensusMemory()

        with get_db_connection(memory.db_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT d.data, c.topic, c.conclusion
                FROM dissent d
                LEFT JOIN consensus c ON d.debate_id = c.id
                ORDER BY d.timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

        dissents = []
        for row in rows:
            try:
                from aragora.memory.consensus import DissentRecord
                record = DissentRecord.from_dict(json.loads(row[0]))
                topic_name = row[1] or "Unknown topic"
                majority_view = row[2] or "No consensus recorded"

                dissents.append({
                    "topic": topic_name,
                    "majority_view": majority_view,
                    "dissenting_view": record.content,
                    "dissenting_agent": record.agent_id,
                    "confidence": record.confidence,
                    "reasoning": record.reasoning if record.reasoning else None,
                })
            except Exception as e:
                logger.debug(f"Failed to parse dissent record: {e}")

        return json_response({"dissents": dissents})

    @ttl_cache(ttl_seconds=CACHE_TTL_CONTRARIAN_VIEWS, key_prefix="contrarian_views", skip_first=True)
    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("contrarian views retrieval")
    def _get_contrarian_views(
        self, topic: Optional[str], domain: Optional[str], limit: int
    ) -> HandlerResult:
        """Get historical contrarian/dissenting views."""
        memory = ConsensusMemory()

        if topic and DissentRetriever is not None:
            retriever = DissentRetriever(memory)
            records = retriever.find_contrarian_views(topic, domain=domain, limit=limit)
        else:
            with get_db_connection(memory.db_path) as conn:
                cursor = conn.cursor()
                query = """
                    SELECT data FROM dissent
                    WHERE dissent_type IN ('fundamental_disagreement', 'alternative_approach')
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()

            records = []
            for row in rows:
                try:
                    from aragora.memory.consensus import DissentRecord
                    records.append(DissentRecord.from_dict(json.loads(row[0])))
                except Exception as e:
                    logger.debug(f"Failed to parse contrarian view record: {e}")

        return json_response({
            "views": [
                {
                    "agent": r.agent_id,
                    "position": r.content,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "debate_id": r.debate_id,
                }
                for r in records
            ],
        })

    @ttl_cache(ttl_seconds=CACHE_TTL_RISK_WARNINGS, key_prefix="risk_warnings", skip_first=True)
    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("risk warnings retrieval")
    def _get_risk_warnings(
        self, topic: Optional[str], domain: Optional[str], limit: int
    ) -> HandlerResult:
        """Get risk warnings and edge case concerns."""
        memory = ConsensusMemory()

        if topic and DissentRetriever is not None:
            retriever = DissentRetriever(memory)
            records = retriever.find_risk_warnings(topic, domain=domain, limit=limit)
        else:
            with get_db_connection(memory.db_path) as conn:
                cursor = conn.cursor()
                query = """
                    SELECT data FROM dissent
                    WHERE dissent_type IN ('risk_warning', 'edge_case_concern')
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()

            records = []
            for row in rows:
                try:
                    from aragora.memory.consensus import DissentRecord
                    records.append(DissentRecord.from_dict(json.loads(row[0])))
                except Exception as e:
                    logger.debug(f"Failed to parse risk warning record: {e}")

        def infer_severity(confidence: float, dissent_type: str) -> str:
            if dissent_type == "risk_warning":
                if confidence >= 0.8:
                    return "critical"
                elif confidence >= 0.6:
                    return "high"
                elif confidence >= 0.4:
                    return "medium"
            return "low"

        return json_response({
            "warnings": [
                {
                    "domain": r.metadata.get("domain", "general"),
                    "risk_type": r.dissent_type.value.replace("_", " ").title(),
                    "severity": infer_severity(r.confidence, r.dissent_type.value),
                    "description": r.content,
                    "mitigation": r.rebuttal if r.rebuttal else None,
                    "detected_at": r.timestamp.isoformat(),
                }
                for r in records
            ],
        })

    @require_feature(lambda: CONSENSUS_MEMORY_AVAILABLE, "Consensus memory")
    @handle_errors("domain history retrieval")
    def _get_domain_history(self, domain: str, limit: int) -> HandlerResult:
        """Get consensus history for a domain."""
        memory = ConsensusMemory()
        records = memory.get_domain_consensus_history(domain, limit=limit)
        return json_response({
            "domain": domain,
            "history": [r.to_dict() for r in records],
            "count": len(records),
        })
