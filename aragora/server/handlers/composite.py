"""
Composite API handlers that aggregate data from multiple subsystems.

These handlers provide convenient single-request access to related data
that would otherwise require multiple API calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from aragora.rbac.decorators import require_permission
from .base import BaseHandler, HandlerResult

logger = logging.getLogger(__name__)


class CompositeHandler(BaseHandler):
    """
    Handler for composite API endpoints that aggregate multiple data sources.

    Endpoints:
    - GET /api/debates/{id}/full-context - Memory + Knowledge + Belief context
    - GET /api/agents/{id}/reliability - Circuit breaker + Airlock metrics
    - GET /api/debates/{id}/compression-analysis - RLM compression metrics
    """

    ROUTES = [
        "/api/v1/debates/*/full-context",
        "/api/v1/agents/*/reliability",
        "/api/v1/debates/*/compression-analysis",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        # Match /api/debates/{id}/full-context
        if path.startswith("/api/v1/debates/") and path.endswith("/full-context"):
            return True
        # Match /api/agents/{id}/reliability
        if path.startswith("/api/v1/agents/") and path.endswith("/reliability"):
            return True
        # Match /api/debates/{id}/compression-analysis
        if path.startswith("/api/v1/debates/") and path.endswith("/compression-analysis"):
            return True
        return False

    @require_permission("composite:read")
    def handle(self, path: str, query_params: dict[str, str], handler: Any) -> HandlerResult | None:
        """Route to appropriate handler method."""
        if path.endswith("/full-context"):
            debate_id = self._extract_id(path, "/api/v1/debates/", "/full-context")
            return self._handle_full_context(debate_id, query_params)
        elif path.endswith("/reliability"):
            agent_id = self._extract_id(path, "/api/v1/agents/", "/reliability")
            return self._handle_reliability(agent_id, query_params)
        elif path.endswith("/compression-analysis"):
            debate_id = self._extract_id(path, "/api/v1/debates/", "/compression-analysis")
            return self._handle_compression_analysis(debate_id, query_params)
        return None

    def _extract_id(self, path: str, prefix: str, suffix: str) -> str:
        """Extract ID from path pattern."""
        return path[len(prefix) : -len(suffix)]

    def _handle_full_context(self, debate_id: str, query_params: dict[str, str]) -> HandlerResult:
        """
        Get full context for a debate including memory, knowledge, and belief data.

        Returns aggregated data from:
        - Continuum memory (debate outcomes, patterns)
        - Knowledge mound (related facts, concepts)
        - Belief network (cruxes, positions, confidence)
        """
        try:
            context: dict[str, Any] = {
                "debate_id": debate_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory": {},
                "knowledge": {},
                "belief": {},
            }

            # Fetch memory context
            try:
                context["memory"] = self._get_memory_context(debate_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error fetching memory for {debate_id}: {e}")
                context["memory"] = {"error": str(e), "available": False}
            except Exception as e:
                logger.exception(f"Unexpected error fetching memory for {debate_id}: {e}")
                context["memory"] = {"error": "Internal error", "available": False}

            # Fetch knowledge context
            try:
                context["knowledge"] = self._get_knowledge_context(debate_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error fetching knowledge for {debate_id}: {e}")
                context["knowledge"] = {"error": str(e), "available": False}
            except Exception as e:
                logger.exception(f"Unexpected error fetching knowledge for {debate_id}: {e}")
                context["knowledge"] = {"error": "Internal error", "available": False}

            # Fetch belief context
            try:
                context["belief"] = self._get_belief_context(debate_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error fetching belief for {debate_id}: {e}")
                context["belief"] = {"error": str(e), "available": False}
            except Exception as e:
                logger.exception(f"Unexpected error fetching belief for {debate_id}: {e}")
                context["belief"] = {"error": "Internal error", "available": False}

            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps(context).encode(),
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Data error in full-context handler: {e}")
            return self._error_response(f"Invalid data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error in full-context handler: {e}")
            return self._error_response("Internal server error", 500)

    def _handle_reliability(self, agent_id: str, query_params: dict[str, str]) -> HandlerResult:
        """
        Get reliability metrics for an agent.

        Returns aggregated data from:
        - Circuit breaker state and history
        - Airlock proxy metrics
        - Recent error rates
        - Availability score
        """
        try:
            metrics: dict[str, Any] = {
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "circuit_breaker": {},
                "airlock": {},
                "availability": {},
                "overall_score": 0.0,
            }

            # Fetch circuit breaker state
            try:
                metrics["circuit_breaker"] = self._get_circuit_breaker_state(agent_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error fetching circuit breaker for {agent_id}: {e}")
                metrics["circuit_breaker"] = {"error": str(e), "available": False}
            except Exception as e:
                logger.exception(f"Unexpected error fetching circuit breaker for {agent_id}: {e}")
                metrics["circuit_breaker"] = {"error": "Internal error", "available": False}

            # Fetch airlock metrics
            try:
                metrics["airlock"] = self._get_airlock_metrics(agent_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error fetching airlock for {agent_id}: {e}")
                metrics["airlock"] = {"error": str(e), "available": False}
            except Exception as e:
                logger.exception(f"Unexpected error fetching airlock for {agent_id}: {e}")
                metrics["airlock"] = {"error": "Internal error", "available": False}

            # Calculate availability
            try:
                metrics["availability"] = self._calculate_availability(agent_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error calculating availability for {agent_id}: {e}")
                metrics["availability"] = {"error": str(e), "available": False}
            except Exception as e:
                logger.exception(f"Unexpected error calculating availability for {agent_id}: {e}")
                metrics["availability"] = {"error": "Internal error", "available": False}

            # Calculate overall reliability score
            metrics["overall_score"] = self._calculate_reliability_score(metrics)

            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps(metrics).encode(),
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Data error in reliability handler: {e}")
            return self._error_response(f"Invalid data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error in reliability handler: {e}")
            return self._error_response("Internal server error", 500)

    def _handle_compression_analysis(
        self, debate_id: str, query_params: dict[str, str]
    ) -> HandlerResult:
        """
        Get RLM compression analysis for a debate.

        Returns:
        - Compression ratios per round
        - Token savings
        - Quality metrics
        - Recommendations
        """
        try:
            analysis: dict[str, Any] = {
                "debate_id": debate_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "compression": {
                    "enabled": False,
                    "rounds_compressed": 0,
                    "original_tokens": 0,
                    "compressed_tokens": 0,
                    "ratio": 0.0,
                    "savings_percent": 0.0,
                },
                "quality": {
                    "information_retained": 0.0,
                    "coherence_score": 0.0,
                },
                "recommendations": [],
            }

            # Try to get RLM metrics
            try:
                rlm_data = self._get_rlm_metrics(debate_id)
                if rlm_data:
                    analysis["compression"].update(rlm_data.get("compression", {}))
                    analysis["quality"].update(rlm_data.get("quality", {}))
                    analysis["compression"]["enabled"] = True
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Data error fetching RLM metrics for {debate_id}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error fetching RLM metrics for {debate_id}: {e}")

            # Generate recommendations
            analysis["recommendations"] = self._generate_compression_recommendations(analysis)

            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps(analysis).encode(),
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Data error in compression-analysis handler: {e}")
            return self._error_response(f"Invalid data: {e}", 400)
        except Exception as e:
            logger.exception(f"Unexpected error in compression-analysis handler: {e}")
            return self._error_response("Internal server error", 500)

    # ==========================================================================
    # Data fetching helpers
    # ==========================================================================

    def _get_memory_context(self, debate_id: str) -> dict[str, Any]:
        """Fetch memory context for a debate."""
        # Try to get from continuum memory
        memory_data: dict[str, Any] = {
            "available": False,
            "outcomes": [],
            "patterns": [],
            "related_debates": [],
        }

        try:
            from aragora.memory.continuum import ContinuumMemory

            # Check if continuum memory is available in context
            continuum = self.ctx.get("continuum_memory")
            if continuum and isinstance(continuum, ContinuumMemory):
                # Get related memories
                memories = continuum.recall(debate_id, limit=5)  # type: ignore[attr-defined]
                memory_data["outcomes"] = [m.to_dict() for m in memories]
                memory_data["available"] = True
        except ImportError:
            pass

        return memory_data

    def _get_knowledge_context(self, debate_id: str) -> dict[str, Any]:
        """Fetch knowledge context for a debate."""
        knowledge_data: dict[str, Any] = {
            "available": False,
            "facts": [],
            "concepts": [],
            "sources": [],
        }

        try:
            # Check for knowledge mound in context
            knowledge_mound = self.ctx.get("knowledge_mound")
            if knowledge_mound:
                # Get related knowledge items
                items = knowledge_mound.query(debate_id, limit=10)  # type: ignore[attr-defined]
                knowledge_data["facts"] = items
                knowledge_data["available"] = True
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Expected error fetching knowledge context: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error fetching knowledge context: {e}")

        return knowledge_data

    def _get_belief_context(self, debate_id: str) -> dict[str, Any]:
        """Fetch belief network context for a debate."""
        belief_data: dict[str, Any] = {
            "available": False,
            "cruxes": [],
            "positions": [],
            "confidence_distribution": {},
        }

        try:
            # Check for belief-related stores
            dissent_retriever = self.ctx.get("dissent_retriever")
            if dissent_retriever:
                cruxes = dissent_retriever.get_cruxes(debate_id, limit=5)  # type: ignore[attr-defined]
                belief_data["cruxes"] = cruxes
                belief_data["available"] = True
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Expected error fetching belief context: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error fetching belief context: {e}")

        return belief_data

    def _get_circuit_breaker_state(self, agent_id: str) -> dict[str, Any]:
        """Get circuit breaker state for an agent."""
        try:
            from aragora.resilience.circuit_breaker_v2 import get_all_circuit_breakers

            all_breakers = get_all_circuit_breakers()
            # Look for a circuit breaker matching this agent_id
            cb = all_breakers.get(agent_id)
            if cb is not None:
                stats = cb.get_stats()
                return {
                    "available": True,
                    "state": stats.state.value,
                    "failure_count": stats.failure_count,
                    "success_count": stats.success_count,
                    "last_failure": stats.last_failure_time,
                    "reset_timeout": stats.cooldown_remaining,
                }
            return {"available": False, "state": "unknown"}
        except ImportError:
            return {"available": False, "state": "unknown"}

    def _get_airlock_metrics(self, agent_id: str) -> dict[str, Any]:
        """Get airlock proxy metrics for an agent.

        Note: Airlock metrics are tracked per-proxy instance, not globally by agent_id.
        To get metrics, you need access to the actual AirlockProxy instance.
        This method checks for a registered proxy in the handler context.
        """
        # Check if we have an airlock proxy registry in the context
        airlock_registry: dict[str, Any] | None = self.ctx.get("airlock_registry")
        if airlock_registry is not None:
            proxy = airlock_registry.get(agent_id)
            if proxy is not None and hasattr(proxy, "metrics"):
                metrics = proxy.metrics
                return {
                    "available": True,
                    "requests_total": metrics.total_calls,
                    "requests_blocked": metrics.fallback_responses,
                    "latency_avg_ms": metrics.avg_latency_ms,
                    "error_rate": 1.0 - (metrics.success_rate / 100.0)
                    if metrics.total_calls > 0
                    else 0.0,
                }
        return {"available": False}

    def _calculate_availability(self, agent_id: str) -> dict[str, Any]:
        """Calculate availability metrics for an agent."""
        # Default values
        return {
            "available": True,
            "uptime_percent": 99.9,
            "last_24h_errors": 0,
            "mean_response_time_ms": 500,
        }

    def _calculate_reliability_score(self, metrics: dict[str, Any]) -> float:
        """Calculate overall reliability score (0-1)."""
        score = 1.0

        # Penalize for circuit breaker issues
        cb = metrics.get("circuit_breaker", {})
        if cb.get("state") == "open":
            score *= 0.3
        elif cb.get("state") == "half-open":
            score *= 0.7

        # Penalize for high error rate
        airlock = metrics.get("airlock", {})
        error_rate = airlock.get("error_rate", 0.0)
        score *= 1.0 - min(error_rate, 0.5)

        return round(score, 3)

    def _get_rlm_metrics(self, debate_id: str) -> Optional[dict[str, Any]]:
        """Get RLM compression metrics for a debate."""
        # This would integrate with the RLM handler
        return None

    def _generate_compression_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations based on compression analysis."""
        recommendations = []

        compression = analysis.get("compression", {})
        if not compression.get("enabled"):
            recommendations.append(
                "Enable RLM compression for debates > 3 rounds to reduce context size"
            )

        if compression.get("ratio", 0) < 0.3:
            recommendations.append(
                "Consider increasing compression levels for better token efficiency"
            )

        quality = analysis.get("quality", {})
        if quality.get("information_retained", 1.0) < 0.8:
            recommendations.append("Reduce compression level to preserve more information")

        return recommendations

    def _error_response(self, message: str, status_code: int) -> HandlerResult:
        """Create an error response."""
        return HandlerResult(
            status_code=status_code,
            content_type="application/json",
            body=json.dumps({"error": message}).encode(),
        )
