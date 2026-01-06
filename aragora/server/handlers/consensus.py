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
import sqlite3
from typing import Optional

from aragora.config import DB_TIMEOUT_SECONDS
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_float_param,
    ttl_cache,
)

logger = logging.getLogger(__name__)

# Lazy import flags
CONSENSUS_MEMORY_AVAILABLE = False
ConsensusMemory = None
DissentRetriever = None

try:
    from aragora.memory.consensus import ConsensusMemory as _CM, DissentRetriever as _DR
    ConsensusMemory = _CM
    DissentRetriever = _DR
    CONSENSUS_MEMORY_AVAILABLE = True
except ImportError:
    pass

from aragora.server.error_utils import safe_error_message as _safe_error_message


class ConsensusHandler(BaseHandler):
    """Handler for consensus memory endpoints."""

    ROUTES = [
        "/api/consensus/similar",
        "/api/consensus/settled",
        "/api/consensus/stats",
        "/api/consensus/dissents",
        "/api/consensus/contrarian-views",
        "/api/consensus/risk-warnings",
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
            topic = query_params.get('topic', '')
            if not topic or len(topic) > 500:
                return error_response("Topic required (max 500 chars)", 400)
            limit = get_int_param(query_params, 'limit', 5)
            return self._get_similar_debates(topic.strip()[:500], min(limit, 20))

        if path == "/api/consensus/settled":
            min_confidence = get_float_param(query_params, 'min_confidence', 0.8)
            min_confidence = max(0.0, min(1.0, min_confidence))
            limit = get_int_param(query_params, 'limit', 20)
            return self._get_settled_topics(min_confidence, min(limit, 100))

        if path == "/api/consensus/stats":
            return self._get_consensus_stats()

        if path == "/api/consensus/dissents":
            topic = query_params.get('topic', '')
            if topic and len(topic) > 500:
                topic = topic[:500]
            domain = query_params.get('domain')
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_recent_dissents(
                topic.strip() if topic else None,
                domain,
                min(limit, 50)
            )

        if path == "/api/consensus/contrarian-views":
            topic = query_params.get('topic', '')
            if topic and len(topic) > 500:
                topic = topic[:500]
            domain = query_params.get('domain')
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_contrarian_views(
                topic.strip() if topic else None,
                domain,
                min(limit, 50)
            )

        if path == "/api/consensus/risk-warnings":
            topic = query_params.get('topic', '')
            if topic and len(topic) > 500:
                topic = topic[:500]
            domain = query_params.get('domain')
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_risk_warnings(
                topic.strip() if topic else None,
                domain,
                min(limit, 50)
            )

        if path.startswith("/api/consensus/domain/"):
            # Extract domain from path
            parts = path.split('/')
            if len(parts) >= 5:
                domain = parts[4]
                limit = get_int_param(query_params, 'limit', 50)
                return self._get_domain_history(domain, min(limit, 200))

        return None

    @ttl_cache(ttl_seconds=240, key_prefix="consensus_similar", skip_first=True)
    def _get_similar_debates(self, topic: str, limit: int) -> HandlerResult:
        """Find debates similar to a topic."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        if not topic:
            return error_response("topic parameter required", 400)

        try:
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
        except Exception as e:
            return error_response(_safe_error_message(e, "similar_topics"), 500)

    @ttl_cache(ttl_seconds=600, key_prefix="consensus_settled", skip_first=True)
    def _get_settled_topics(self, min_confidence: float, limit: int) -> HandlerResult:
        """Get high-confidence settled topics."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        try:
            memory = ConsensusMemory()
            with sqlite3.connect(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
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
        except Exception as e:
            return error_response(_safe_error_message(e, "settled_topics"), 500)

    @ttl_cache(ttl_seconds=600, key_prefix="consensus_stats", skip_first=True)
    def _get_consensus_stats(self) -> HandlerResult:
        """Get consensus memory statistics."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        try:
            memory = ConsensusMemory()
            raw_stats = memory.get_statistics()

            with sqlite3.connect(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM consensus WHERE confidence >= 0.7")
                row = cursor.fetchone()
                high_confidence_count = row[0] if row else 0
                cursor.execute("SELECT AVG(confidence) FROM consensus")
                avg_row = cursor.fetchone()
                avg_confidence = avg_row[0] if avg_row and avg_row[0] else 0.0

            return json_response({
                "total_topics": raw_stats.get("total_consensus", 0),
                "high_confidence_count": high_confidence_count,
                "domains": list(raw_stats.get("by_domain", {}).keys()),
                "avg_confidence": round(avg_confidence, 3),
                "total_dissents": raw_stats.get("total_dissents", 0),
                "by_strength": raw_stats.get("by_strength", {}),
                "by_domain": raw_stats.get("by_domain", {}),
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "consensus_stats"), 500)

    def _get_recent_dissents(
        self, topic: Optional[str], domain: Optional[str], limit: int
    ) -> HandlerResult:
        """Get recent dissents, optionally filtered by topic."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        try:
            memory = ConsensusMemory()

            with sqlite3.connect(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
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
        except Exception as e:
            return error_response(_safe_error_message(e, "recent_dissents"), 500)

    def _get_contrarian_views(
        self, topic: Optional[str], domain: Optional[str], limit: int
    ) -> HandlerResult:
        """Get historical contrarian/dissenting views."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        try:
            memory = ConsensusMemory()

            if topic and DissentRetriever is not None:
                retriever = DissentRetriever(memory)
                records = retriever.find_contrarian_views(topic, domain=domain, limit=limit)
            else:
                with sqlite3.connect(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
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
        except Exception as e:
            return error_response(_safe_error_message(e, "contrarian_views"), 500)

    def _get_risk_warnings(
        self, topic: Optional[str], domain: Optional[str], limit: int
    ) -> HandlerResult:
        """Get risk warnings and edge case concerns."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        try:
            memory = ConsensusMemory()

            if topic and DissentRetriever is not None:
                retriever = DissentRetriever(memory)
                records = retriever.find_risk_warnings(topic, domain=domain, limit=limit)
            else:
                with sqlite3.connect(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
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
        except Exception as e:
            return error_response(_safe_error_message(e, "risk_warnings"), 500)

    def _get_domain_history(self, domain: str, limit: int) -> HandlerResult:
        """Get consensus history for a domain."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            return error_response("Consensus memory not available", 503)

        try:
            memory = ConsensusMemory()
            records = memory.get_domain_consensus_history(domain, limit=limit)
            return json_response({
                "domain": domain,
                "history": [r.to_dict() for r in records],
                "count": len(records),
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "domain_history"), 500)
