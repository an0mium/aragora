"""
Knowledge Mound event handler mixin for CrossSubscriberManager.

Handles bidirectional event flow between Knowledge Mound and other subsystems:
- Mound ↔ Memory: Retrieval patterns, consensus storage
- Mound ↔ Belief: Provenance tracking
- Mound ↔ RLM: Compression strategies
- Mound ↔ ELO: Performance metrics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.events.types import StreamEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MoundHandlersMixin:
    """Mixin providing Knowledge Mound event handlers."""

    # These will be set by the main class
    stats: dict
    retry_handler: Any
    circuit_breaker: Any
    _culture_cache: dict
    _culture_cache_ttl: float
    _staleness_debounce: dict
    _staleness_debounce_seconds: float

    def _is_km_handler_enabled(self, handler_name: str) -> bool:
        """Check if a KM handler is enabled (defined in main class)."""
        raise NotImplementedError

    def _handle_mound_to_memory(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → Memory events.

        Syncs knowledge nodes to memory for retrieval.
        """
        if not self._is_km_handler_enabled("mound_to_memory"):
            return

        try:
            from aragora.observability.metrics.slo import check_and_record_slo
        except ImportError:
            check_and_record_slo = None

        import time

        start = time.time()

        try:
            data = event.data
            node_id = data.get("node_id")
            content = data.get("content")
            metadata = data.get("metadata", {})
            workspace = data.get("workspace", "default")

            if not node_id or not content:
                logger.warning("Mound → Memory: Missing node_id or content")
                return

            # Try to sync to memory continuum
            try:
                from aragora.memory.continuum import ContinuumMemory

                memory = ContinuumMemory()
                memory.store(  # type: ignore[unused-coroutine]
                    key=f"km:{node_id}",
                    value=content,
                    metadata={
                        "source": "knowledge_mound",
                        "workspace": workspace,
                        **metadata,
                    },
                    tier="medium",  # Cross-session memory
                )
                logger.debug(f"Synced KM node {node_id} to memory continuum")
            except Exception as e:
                logger.debug(f"Memory sync unavailable: {e}")

            self.stats["mound_to_memory"]["events"] += 1

            # Record SLO if available
            if check_and_record_slo:
                latency_ms = (time.time() - start) * 1000
                check_and_record_slo("km_mound_to_memory", latency_ms)

        except Exception as e:
            logger.error(f"Mound → Memory handler error: {e}")
            self.stats["mound_to_memory"]["errors"] += 1

    def _handle_memory_to_mound(self, event: StreamEvent) -> None:
        """Handle Memory → Knowledge Mound events.

        Informs KM about memory access patterns for revalidation.
        """
        if not self._is_km_handler_enabled("memory_to_mound"):
            return

        try:
            from aragora.observability.metrics.slo import check_and_record_slo
        except ImportError:
            check_and_record_slo = None

        import time

        start = time.time()

        try:
            data = event.data
            memory_key = data.get("key")
            access_type = data.get("access_type", "read")
            hit = data.get("hit", True)

            if not memory_key:
                return

            # Extract KM node ID if this is a KM-sourced memory
            if memory_key.startswith("km:"):
                node_id = memory_key[3:]  # Remove "km:" prefix

                try:
                    from aragora.knowledge.mound_core import KnowledgeMound

                    mound = KnowledgeMound()
                    if hasattr(mound, "record_access"):
                        mound.record_access(node_id, access_type=access_type, hit=hit)
                        logger.debug(f"Recorded KM access for node {node_id}")
                except Exception as e:
                    logger.debug(f"KM access recording unavailable: {e}")

            self.stats["memory_to_mound"]["events"] += 1

            # Record SLO
            if check_and_record_slo:
                latency_ms = (time.time() - start) * 1000
                check_and_record_slo("km_memory_to_mound", latency_ms)

        except Exception as e:
            logger.error(f"Memory → Mound handler error: {e}")
            self.stats["memory_to_mound"]["errors"] += 1

    def _handle_belief_to_mound(self, event: StreamEvent) -> None:
        """Handle Belief Network → Knowledge Mound events.

        Syncs provenance and claim data to KM for cross-debate learning.
        """
        if not self._is_km_handler_enabled("belief_to_mound"):
            return

        try:
            data = event.data
            claim_id = data.get("claim_id")
            claim_text = data.get("claim_text")
            confidence = data.get("confidence", 0.5)
            sources = data.get("sources", [])
            workspace = data.get("workspace", "default")

            if not claim_id or not claim_text:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "ingest_belief"):
                    mound.ingest_belief(
                        claim_id=claim_id,
                        claim_text=claim_text,
                        confidence=confidence,
                        sources=sources,
                    )
                    logger.debug(f"Ingested belief {claim_id} to KM")
            except Exception as e:
                logger.debug(f"KM belief ingestion unavailable: {e}")

            self.stats["belief_to_mound"]["events"] += 1

        except Exception as e:
            logger.error(f"Belief → Mound handler error: {e}")
            self.stats["belief_to_mound"]["errors"] += 1

    def _handle_mound_to_belief(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → Belief Network events.

        Provides historical claim context for provenance tracking.
        """
        if not self._is_km_handler_enabled("mound_to_belief"):
            return

        try:
            data = event.data
            query = data.get("query")
            debate_id = data.get("debate_id")
            workspace = data.get("workspace", "default")

            if not query:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "find_related_claims"):
                    related = mound.find_related_claims(query, limit=5)
                    if related and debate_id:
                        # Store for belief network to pick up
                        logger.debug(f"Found {len(related)} related claims for debate {debate_id}")
            except Exception as e:
                logger.debug(f"KM claim lookup unavailable: {e}")

            self.stats["mound_to_belief"]["events"] += 1

        except Exception as e:
            logger.error(f"Mound → Belief handler error: {e}")
            self.stats["mound_to_belief"]["errors"] += 1

    def _handle_rlm_to_mound(self, event: StreamEvent) -> None:
        """Handle RLM → Knowledge Mound events.

        Syncs compression artifacts and retrieval patterns.
        """
        if not self._is_km_handler_enabled("rlm_to_mound"):
            return

        try:
            data = event.data
            content_id = data.get("content_id")
            compression_level = data.get("compression_level")
            retrieval_count = data.get("retrieval_count", 0)
            workspace = data.get("workspace", "default")

            if not content_id:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "update_rlm_metadata"):
                    mound.update_rlm_metadata(
                        content_id=content_id,
                        compression_level=compression_level,
                        retrieval_count=retrieval_count,
                    )
            except Exception as e:
                logger.debug(f"KM RLM metadata update unavailable: {e}")

            self.stats["rlm_to_mound"]["events"] += 1

        except Exception as e:
            logger.error(f"RLM → Mound handler error: {e}")
            self.stats["rlm_to_mound"]["errors"] += 1

    def _handle_mound_to_rlm(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → RLM events.

        Provides context hints for compression strategies.
        """
        if not self._is_km_handler_enabled("mound_to_rlm"):
            return

        try:
            data = event.data
            content_type = data.get("content_type")
            workspace = data.get("workspace", "default")

            if not content_type:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "get_compression_hints"):
                    hints = mound.get_compression_hints(content_type)
                    if hints:
                        logger.debug(f"Retrieved compression hints for {content_type}")
            except Exception as e:
                logger.debug(f"KM compression hints unavailable: {e}")

            self.stats["mound_to_rlm"]["events"] += 1

        except Exception as e:
            logger.error(f"Mound → RLM handler error: {e}")
            self.stats["mound_to_rlm"]["errors"] += 1

    def _handle_elo_to_mound(self, event: StreamEvent) -> None:
        """Handle ELO → Knowledge Mound events.

        Syncs agent performance metrics for historical tracking.
        """
        if not self._is_km_handler_enabled("elo_to_mound"):
            return

        try:
            data = event.data
            agent_name = data.get("agent_name")
            new_rating = data.get("new_rating")
            rating_change = data.get("rating_change", 0)
            domain = data.get("domain", "general")
            workspace = data.get("workspace", "default")

            if not agent_name or new_rating is None:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "record_elo_update"):
                    mound.record_elo_update(
                        agent_name=agent_name,
                        new_rating=new_rating,
                        rating_change=rating_change,
                        domain=domain,
                    )
                    logger.debug(f"Recorded ELO update for {agent_name} in KM")
            except Exception as e:
                logger.debug(f"KM ELO recording unavailable: {e}")

            self.stats["elo_to_mound"]["events"] += 1

        except Exception as e:
            logger.error(f"ELO → Mound handler error: {e}")
            self.stats["elo_to_mound"]["errors"] += 1

    def _handle_mound_to_team_selection(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → Team Selection events.

        Provides historical performance data for team composition.
        """
        if not self._is_km_handler_enabled("mound_to_team_selection"):
            return

        try:
            data = event.data
            task_type = data.get("task_type")
            candidate_agents = data.get("candidate_agents", [])
            workspace = data.get("workspace", "default")

            if not task_type or not candidate_agents:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "get_agent_history"):
                    for agent in candidate_agents:
                        history = mound.get_agent_history(agent, task_type=task_type)
                        if history:
                            logger.debug(f"Retrieved history for {agent} on {task_type}")
            except Exception as e:
                logger.debug(f"KM agent history unavailable: {e}")

            self.stats["mound_to_team_selection"]["events"] += 1

        except Exception as e:
            logger.error(f"Mound → Team Selection handler error: {e}")
            self.stats["mound_to_team_selection"]["errors"] += 1

    def _handle_insight_to_mound(self, event: StreamEvent) -> None:
        """Handle Insight → Knowledge Mound events.

        Syncs debate insights for cross-debate learning.
        """
        if not self._is_km_handler_enabled("insight_to_mound"):
            return

        try:
            data = event.data
            insight_type = data.get("type")
            content = data.get("content")
            debate_id = data.get("debate_id")
            workspace = data.get("workspace", "default")

            if not insight_type or not content:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "ingest_insight"):
                    mound.ingest_insight(
                        insight_type=insight_type,
                        content=content,
                        debate_id=debate_id,
                    )
                    logger.debug(f"Ingested {insight_type} insight to KM")
            except Exception as e:
                logger.debug(f"KM insight ingestion unavailable: {e}")

            self.stats["insight_to_mound"]["events"] += 1

        except Exception as e:
            logger.error(f"Insight → Mound handler error: {e}")
            self.stats["insight_to_mound"]["errors"] += 1

    def _handle_flip_to_mound(self, event: StreamEvent) -> None:
        """Handle Position Flip → Knowledge Mound events.

        Records agent position changes for tracking persuasion patterns.
        """
        if not self._is_km_handler_enabled("flip_to_mound"):
            return

        try:
            data = event.data
            agent_name = data.get("agent")
            from_position = data.get("from_position")
            to_position = data.get("to_position")
            reason = data.get("reason")
            debate_id = data.get("debate_id")
            workspace = data.get("workspace", "default")

            if not agent_name or not debate_id:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "record_position_flip"):
                    mound.record_position_flip(
                        agent_name=agent_name,
                        from_position=from_position,
                        to_position=to_position,
                        reason=reason,
                        debate_id=debate_id,
                    )
            except Exception as e:
                logger.debug(f"KM position flip recording unavailable: {e}")

            self.stats["flip_to_mound"]["events"] += 1

        except Exception as e:
            logger.error(f"Flip → Mound handler error: {e}")
            self.stats["flip_to_mound"]["errors"] += 1

    def _handle_mound_to_trickster(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → Trickster events.

        Provides historical hollow consensus patterns.
        """
        if not self._is_km_handler_enabled("mound_to_trickster"):
            return

        try:
            data = event.data
            consensus_topic = data.get("topic")
            workspace = data.get("workspace", "default")

            if not consensus_topic:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "get_hollow_consensus_patterns"):
                    patterns = mound.get_hollow_consensus_patterns(consensus_topic)
                    if patterns:
                        logger.debug(f"Found {len(patterns)} hollow consensus patterns for topic")
            except Exception as e:
                logger.debug(f"KM hollow consensus patterns unavailable: {e}")

            self.stats["mound_to_trickster"]["events"] += 1

        except Exception as e:
            logger.error(f"Mound → Trickster handler error: {e}")
            self.stats["mound_to_trickster"]["errors"] += 1
