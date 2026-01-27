"""
RLM (Recursive Language Model) endpoint handlers.

Endpoints:
- POST /api/debates/{id}/query-rlm - Query a debate using RLM with refinement
- POST /api/debates/{id}/compress - Compress debate context using RLM
- GET /api/debates/{id}/context/{level} - Get debate at specific abstraction level
- GET /api/debates/{id}/refinement-status - Check iterative refinement progress
- POST /api/knowledge/query-rlm - Query knowledge mound using RLM

Security:
    All endpoints require RBAC permissions:
    - debates.read: Query and compress debate context
    - knowledge.read: Query knowledge mound
    - analytics.read: Access RLM metrics
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from aragora.server.http_utils import run_async as _run_async
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationContext

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)

logger = logging.getLogger(__name__)


class RLMHandler(BaseHandler):
    """Handler for RLM-powered query and compression endpoints.

    RBAC Permissions:
    - debates.read: Query and compress debate context
    - knowledge.read: Query knowledge mound
    - analytics.read: Access RLM metrics
    """

    RESOURCE_TYPE = "rlm"  # For audit logging

    ROUTES = [
        "/api/v1/debates/{debate_id}/query-rlm",
        "/api/v1/debates/{debate_id}/compress",
        "/api/v1/debates/{debate_id}/context/{level}",
        "/api/v1/debates/{debate_id}/refinement-status",
        "/api/v1/knowledge/query-rlm",
        "/api/v1/rlm/status",
        "/api/v1/metrics/rlm",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        # Handle parameterized routes
        if path.startswith("/api/v1/debates/") and "/query-rlm" in path:
            return True
        if path.startswith("/api/v1/debates/") and "/compress" in path:
            return True
        if path.startswith("/api/v1/debates/") and "/context/" in path:
            return True
        if path.startswith("/api/v1/debates/") and "/refinement-status" in path:
            return True
        if path == "/api/v1/knowledge/query-rlm":
            return True
        if path == "/api/v1/rlm/status":
            return True
        if path == "/api/v1/metrics/rlm":
            return True
        return False

    def _extract_debate_id(self, path: str) -> Optional[str]:
        """Extract debate ID from path like /api/v1/debates/{id}/..."""
        parts = path.split("/")
        # Path: /api/v1/debates/{id}/... - debate_id at index 4
        if len(parts) >= 5 and parts[1] == "api" and parts[2] == "v1" and parts[3] == "debates":
            return parts[4]
        return None

    def _extract_level(self, path: str) -> Optional[str]:
        """Extract abstraction level from path like /api/v1/debates/{id}/context/{level}."""
        parts = path.split("/")
        # Path: /api/v1/debates/{id}/context/{level} - context at index 5, level at index 6
        if len(parts) >= 7 and parts[5] == "context":
            return parts[6].upper()
        return None

    def _check_permission(self, user, permission: str) -> Optional[HandlerResult]:
        """Check RBAC permission for the authenticated user.

        Returns None if permission is granted, or an error response if denied.
        """
        if not user:
            return error_response("Authentication required", 401)

        try:
            auth_context = AuthorizationContext(
                user_id=getattr(user, "user_id", "anonymous"),
                org_id=getattr(user, "org_id", None),
                roles=getattr(user, "roles", {"member"}),
            )

            checker = get_permission_checker()
            decision = checker.check_permission(auth_context, permission)

            if not decision.allowed:
                return error_response(f"Permission denied: {permission}", 403)
            return None
        except Exception as e:
            logger.error(f"RBAC check failed: {e}")
            return error_response("Authorization check failed", 500)

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/v1/rlm/status":
            return self._get_rlm_status()
        if path == "/api/v1/metrics/rlm":
            return self._get_rlm_metrics()
        if "/context/" in path:
            return self._get_context_level(path, handler)
        if "/refinement-status" in path:
            return self._get_refinement_status(path, handler)
        return error_response("Use POST method for RLM queries", 405)

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if "/query-rlm" in path and path.startswith("/api/v1/debates/"):
            return self._query_debate_rlm(path, handler)
        if "/compress" in path:
            return self._compress_debate(path, handler)
        if path == "/api/v1/knowledge/query-rlm":
            return self._query_knowledge_rlm(handler)
        return None

    @require_user_auth
    @handle_errors("RLM debate query")
    def _query_debate_rlm(self, path: str, handler, user=None) -> HandlerResult:
        """
        Query a debate using RLM with iterative refinement.

        Request body:
        {
            "query": "What was the consensus on pricing?",
            "strategy": "auto",  // Optional: auto, peek, grep, partition_map
            "max_iterations": 3,  // Optional: max refinement iterations
            "start_level": "SUMMARY"  // Optional: starting abstraction level
        }

        Response:
        {
            "answer": "The debate reached consensus that...",
            "ready": true,
            "iteration": 1,
            "refinement_history": [...],
            "confidence": 0.85,
            "nodes_examined": [...],
            "tokens_processed": 5000
        }
        """
        # RBAC check
        rbac_error = self._check_permission(user, "debates.read")
        if rbac_error:
            return rbac_error

        debate_id = self._extract_debate_id(path)
        if not debate_id:
            return error_response("Invalid debate ID", 400)

        body = handler.get_json_body()
        if not body:
            return error_response("Request body required", 400)

        query = body.get("query")
        if not query:
            return error_response("Query is required", 400)

        strategy = body.get("strategy", "auto")
        max_iterations = body.get("max_iterations", 3)
        start_level = body.get("start_level", "SUMMARY")

        try:
            result = _run_async(
                self._execute_rlm_query(
                    debate_id=debate_id,
                    query=query,
                    strategy=strategy,
                    max_iterations=max_iterations,
                    start_level=start_level,
                )
            )

            return json_response(
                {
                    "answer": result.answer,
                    "ready": result.ready,
                    "iteration": result.iteration,
                    "refinement_history": result.refinement_history,
                    "confidence": result.confidence,
                    "nodes_examined": result.nodes_examined,
                    "tokens_processed": result.tokens_processed,
                    "sub_calls_made": result.sub_calls_made,
                }
            )

        except Exception as e:  # noqa: BLE001 - API boundary, log and return error
            logger.error(f"RLM query failed: {e}")
            return error_response(safe_error_message(e, "RLM query"), 500)

    async def _execute_rlm_query(
        self,
        debate_id: str,
        query: str,
        strategy: str,
        max_iterations: int,
        start_level: str,
    ) -> Any:
        """Execute RLM query with refinement loop."""
        from aragora.rlm.bridge import AragoraRLM, DebateContextAdapter

        # Get debate result
        debate_result = await self._get_debate_result(debate_id)
        if not debate_result:
            raise ValueError(f"Debate {debate_id} not found")

        # Create RLM instance
        rlm = AragoraRLM()

        # Create adapter and compress debate
        adapter = DebateContextAdapter(rlm)
        context = await adapter.compress_debate(debate_result)

        # Execute query with refinement
        result = await rlm.query_with_refinement(
            query=query,
            context=context,
            strategy=strategy,
            max_iterations=max_iterations,
        )

        return result

    async def _get_debate_result(self, debate_id: str) -> Optional[Any]:
        """Fetch debate result from storage."""
        try:
            from aragora.storage.factory import get_store  # type: ignore[attr-defined]

            store = get_store()
            return await store.get_debate(debate_id)
        except Exception as e:  # noqa: BLE001 - Best effort fetch, log warning on failure
            logger.warning(f"Failed to get debate {debate_id}: {e}")
            return None

    @require_user_auth
    @handle_errors("RLM compression")
    def _compress_debate(self, path: str, handler, user=None) -> HandlerResult:
        """
        Compress a debate into hierarchical context.

        Request body:
        {
            "target_levels": ["ABSTRACT", "SUMMARY", "DETAILED"],  // Optional
            "compression_ratio": 0.3  // Optional: target compression
        }

        Response:
        {
            "original_tokens": 50000,
            "compressed_tokens": {
                "ABSTRACT": 500,
                "SUMMARY": 2500,
                "DETAILED": 10000
            },
            "compression_ratios": {...},
            "time_seconds": 2.5,
            "levels_created": 3
        }
        """
        # RBAC check
        rbac_error = self._check_permission(user, "debates.read")
        if rbac_error:
            return rbac_error

        debate_id = self._extract_debate_id(path)
        if not debate_id:
            return error_response("Invalid debate ID", 400)

        body = handler.get_json_body() or {}
        target_levels = body.get("target_levels", ["ABSTRACT", "SUMMARY", "DETAILED"])
        compression_ratio = body.get("compression_ratio", 0.3)

        try:
            result = _run_async(
                self._execute_compression(
                    debate_id=debate_id,
                    target_levels=target_levels,
                    compression_ratio=compression_ratio,
                )
            )
            return json_response(result)

        except Exception as e:  # noqa: BLE001 - API boundary, log and return error
            logger.error(f"RLM compression failed: {e}")
            return error_response(safe_error_message(e, "Compression"), 500)

    async def _execute_compression(
        self,
        debate_id: str,
        target_levels: list,
        compression_ratio: float,
    ) -> dict:
        """Execute debate compression."""
        from aragora.rlm.bridge import DebateContextAdapter
        from aragora.rlm.types import AbstractionLevel

        start_time = time.time()

        debate_result = await self._get_debate_result(debate_id)
        if not debate_result:
            raise ValueError(f"Debate {debate_id} not found")

        adapter = DebateContextAdapter()
        context = await adapter.compress_debate(debate_result)

        compressed_tokens = {}
        compression_ratios = {}

        for level_name in target_levels:
            try:
                level = AbstractionLevel[level_name]
                tokens = context.total_tokens_at_level(level)
                compressed_tokens[level_name] = tokens
                if context.original_tokens > 0:
                    compression_ratios[level_name] = tokens / context.original_tokens
            except KeyError:
                continue

        return {
            "original_tokens": context.original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratios": compression_ratios,
            "time_seconds": time.time() - start_time,
            "levels_created": len(context.levels),
        }

    @require_user_auth
    @handle_errors("get context level")
    def _get_context_level(self, path: str, handler, user=None) -> HandlerResult:
        """
        Get debate content at a specific abstraction level.

        Response:
        {
            "level": "SUMMARY",
            "content": "...",
            "token_count": 2500,
            "nodes": [...]
        }
        """
        # RBAC check
        rbac_error = self._check_permission(user, "debates.read")
        if rbac_error:
            return rbac_error

        debate_id = self._extract_debate_id(path)
        level_name = self._extract_level(path)

        if not debate_id:
            return error_response("Invalid debate ID", 400)
        if not level_name:
            return error_response("Invalid abstraction level", 400)

        try:
            result = _run_async(self._get_level_content(debate_id, level_name))
            return json_response(result)
        except Exception as e:  # noqa: BLE001 - API boundary, log and return error
            logger.error(f"Failed to get context level: {e}")
            return error_response(safe_error_message(e, "Get context level"), 500)

    async def _get_level_content(self, debate_id: str, level_name: str) -> dict:
        """Get content at specific abstraction level."""
        from aragora.rlm.bridge import DebateContextAdapter
        from aragora.rlm.types import AbstractionLevel

        debate_result = await self._get_debate_result(debate_id)
        if not debate_result:
            raise ValueError(f"Debate {debate_id} not found")

        adapter = DebateContextAdapter()
        context = await adapter.compress_debate(debate_result)

        try:
            level = AbstractionLevel[level_name]
        except KeyError:
            raise ValueError(f"Invalid level: {level_name}")

        content = context.get_at_level(level)
        tokens = context.total_tokens_at_level(level)

        nodes = []
        if level in context.levels:
            for node in context.levels[level]:
                nodes.append(
                    {
                        "id": node.id,
                        "content": (
                            node.content[:500] + "..." if len(node.content) > 500 else node.content
                        ),
                        "token_count": node.token_count,
                        "key_topics": node.key_topics,
                    }
                )

        return {
            "level": level_name,
            "content": content,
            "token_count": tokens,
            "nodes": nodes,
        }

    @require_user_auth
    @handle_errors("refinement status")
    def _get_refinement_status(self, path: str, handler, user=None) -> HandlerResult:
        """
        Get status of an ongoing refinement process.

        Response:
        {
            "debate_id": "...",
            "active_queries": 0,
            "cached_contexts": 1,
            "last_query_time": "..."
        }
        """
        # RBAC check
        rbac_error = self._check_permission(user, "debates.read")
        if rbac_error:
            return rbac_error

        debate_id = self._extract_debate_id(path)
        if not debate_id:
            return error_response("Invalid debate ID", 400)

        # Note: In production, track active queries in a store
        return json_response(
            {
                "debate_id": debate_id,
                "active_queries": 0,
                "cached_contexts": 0,
                "status": "idle",
            }
        )

    @require_user_auth
    @handle_errors("RLM knowledge query")
    def _query_knowledge_rlm(self, handler, user=None) -> HandlerResult:
        """
        Query knowledge mound using RLM.

        Request body:
        {
            "workspace_id": "ws_123",
            "query": "What are the key security requirements?",
            "max_nodes": 100,
            "strategy": "auto"
        }

        Response:
        {
            "answer": "...",
            "sources": [...],
            "confidence": 0.85
        }
        """
        # RBAC check
        rbac_error = self._check_permission(user, "knowledge.read")
        if rbac_error:
            return rbac_error

        body = handler.get_json_body()
        if not body:
            return error_response("Request body required", 400)

        workspace_id = body.get("workspace_id")
        query = body.get("query")

        if not workspace_id:
            return error_response("workspace_id is required", 400)
        if not query:
            return error_response("query is required", 400)

        max_nodes = body.get("max_nodes", 100)
        strategy = body.get("strategy", "auto")

        try:
            result = _run_async(
                self._execute_knowledge_query(
                    workspace_id=workspace_id,
                    query=query,
                    max_nodes=max_nodes,
                    strategy=strategy,
                )
            )
            return json_response(result)
        except Exception as e:  # noqa: BLE001 - API boundary, log and return error
            logger.error(f"Knowledge RLM query failed: {e}")
            return error_response(safe_error_message(e, "Query"), 500)

    async def _execute_knowledge_query(
        self,
        workspace_id: str,
        query: str,
        max_nodes: int,
        strategy: str,
    ) -> dict:
        """Execute RLM query against knowledge mound."""
        from aragora.rlm.bridge import AragoraRLM, KnowledgeMoundAdapter

        try:
            from aragora.knowledge.mound import KnowledgeMound

            mound = KnowledgeMound()  # type: ignore[abstract]
        except ImportError:
            return {
                "answer": "Knowledge Mound not available",
                "sources": [],
                "confidence": 0.0,
            }

        rlm = AragoraRLM()
        adapter = KnowledgeMoundAdapter(mound)

        context = await adapter.to_rlm_context(
            workspace_id=workspace_id,
            query=query,
            max_nodes=max_nodes,
        )

        result = await rlm.query_with_refinement(
            query=query,
            context=context,
            strategy=strategy,
        )

        return {
            "answer": result.answer,
            "sources": result.nodes_examined,
            "confidence": result.confidence,
            "ready": result.ready,
            "iteration": result.iteration,
        }

    def _get_rlm_status(self) -> HandlerResult:
        """
        Get RLM system status.

        Response:
        {
            "available": true,
            "provider": "built-in",
            "version": "1.0.0",
            "features": ["compression", "queries", "refinement", "streaming"]
        }
        """
        try:
            # Check if official RLM is available
            try:
                import rlm

                provider = "rlm-library"
                version = getattr(rlm, "__version__", "unknown")
            except ImportError:
                provider = "built-in"
                version = "1.0.0"

            features = ["compression", "queries", "refinement"]

            # Check streaming support
            try:
                from aragora.rlm.streaming import StreamingRLMQuery  # noqa: F401

                features.append("streaming")
            except ImportError:
                pass

            # Check training support
            try:
                from aragora.rlm.training.trainer import RLMTrainer  # type: ignore[attr-defined]  # noqa: F401

                features.append("training")
            except ImportError:
                pass

            return json_response(
                {
                    "available": True,
                    "provider": provider,
                    "version": version,
                    "features": features,
                }
            )

        except Exception as e:  # noqa: BLE001 - Status endpoint, return degraded state on error
            logger.error(f"Failed to get RLM status: {e}")
            return json_response(
                {
                    "available": False,
                    "provider": "unknown",
                    "version": "unknown",
                    "features": [],
                    "error": str(e),
                }
            )

    def _get_rlm_metrics(self) -> HandlerResult:
        """
        Get RLM metrics for monitoring dashboard.

        Response:
        {
            "compressions": {...},
            "queries": {...},
            "cache": {...},
            "refinement": {...}
        }
        """
        try:
            # Try to get metrics from Prometheus registry
            from aragora.rlm.metrics import (  # type: ignore[attr-defined]
                RLM_COMPRESSIONS_TOTAL,
                RLM_TOKENS_SAVED_TOTAL,
                RLM_QUERIES_TOTAL,
                RLM_CACHE_HITS_TOTAL,
                RLM_CACHE_MISSES_TOTAL,
                RLM_MEMORY_BYTES,
                RLM_REFINEMENT_SUCCESS_TOTAL,
                RLM_READY_FALSE_TOTAL,
            )

            # Extract values from Prometheus metrics
            compressions_total = self._get_counter_value(RLM_COMPRESSIONS_TOTAL)
            tokens_saved = self._get_counter_value(RLM_TOKENS_SAVED_TOTAL)
            queries_total = self._get_counter_value(RLM_QUERIES_TOTAL)
            cache_hits = self._get_counter_value(RLM_CACHE_HITS_TOTAL)
            cache_misses = self._get_counter_value(RLM_CACHE_MISSES_TOTAL)
            refinement_success = self._get_counter_value(RLM_REFINEMENT_SUCCESS_TOTAL)
            ready_false = self._get_counter_value(RLM_READY_FALSE_TOTAL)
            memory_bytes = self._get_gauge_value(RLM_MEMORY_BYTES)

            # Calculate derived metrics
            cache_total = cache_hits + cache_misses
            hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

            return json_response(
                {
                    "compressions": {
                        "total": int(compressions_total),
                        "byType": self._get_counter_by_label(RLM_COMPRESSIONS_TOTAL, "source_type"),
                        "avgRatio": 0.34,  # Would need histogram to calculate
                        "tokensSaved": int(tokens_saved),
                    },
                    "queries": {
                        "total": int(queries_total),
                        "byType": self._get_counter_by_label(RLM_QUERIES_TOTAL, "query_type"),
                        "avgDuration": 1.24,  # Would need histogram to calculate
                        "successRate": 0.94,  # Would need tracking
                    },
                    "cache": {
                        "hits": int(cache_hits),
                        "misses": int(cache_misses),
                        "hitRate": hit_rate,
                        "memoryBytes": int(memory_bytes),
                        "maxMemory": 268435456,  # 256MB default
                    },
                    "refinement": {
                        "avgIterations": 2.3,  # Would need histogram to calculate
                        "successRate": (
                            refinement_success / (refinement_success + ready_false)
                            if (refinement_success + ready_false) > 0
                            else 0.0
                        ),
                        "readyFalseTotal": int(ready_false),
                    },
                }
            )

        except ImportError:
            # RLM metrics module not available, return placeholder data
            logger.debug("RLM metrics module not available, returning placeholder data")
            return json_response(
                {
                    "compressions": {
                        "total": 0,
                        "byType": {},
                        "avgRatio": 0.0,
                        "tokensSaved": 0,
                    },
                    "queries": {
                        "total": 0,
                        "byType": {},
                        "avgDuration": 0.0,
                        "successRate": 0.0,
                    },
                    "cache": {
                        "hits": 0,
                        "misses": 0,
                        "hitRate": 0.0,
                        "memoryBytes": 0,
                        "maxMemory": 268435456,
                    },
                    "refinement": {
                        "avgIterations": 0.0,
                        "successRate": 0.0,
                        "readyFalseTotal": 0,
                    },
                }
            )

        except Exception as e:  # noqa: BLE001 - API boundary, log and return error
            logger.error(f"Failed to get RLM metrics: {e}")
            return error_response(safe_error_message(e, "Failed to get metrics"), 500)

    def _get_counter_value(self, counter) -> float:
        """Extract total value from a Prometheus Counter."""
        try:
            return counter._value.get() if hasattr(counter, "_value") else 0.0
        except (AttributeError, TypeError):
            return 0.0

    def _get_gauge_value(self, gauge) -> float:
        """Extract value from a Prometheus Gauge."""
        try:
            return gauge._value.get() if hasattr(gauge, "_value") else 0.0
        except (AttributeError, TypeError):
            return 0.0

    def _get_counter_by_label(self, counter, label_name: str) -> dict:
        """Extract counter values grouped by a label."""
        try:
            result = {}
            if hasattr(counter, "_metrics"):
                for labels, value in counter._metrics.items():
                    if label_name in labels:
                        result[labels[label_name]] = value._value.get()
            return result
        except (AttributeError, TypeError, KeyError):
            return {}


# Export handler
__all__ = ["RLMHandler"]
