"""
Knowledge Mound bidirectional event handlers.

Handles bidirectional data flow between subsystems and Knowledge Mound:
- Memory ↔ KM: Sync high-importance memories
- Belief ↔ KM: Store converged beliefs, initialize priors
- RLM ↔ KM: Store compression patterns, update priorities
- ELO ↔ KM: Store agent expertise, query domain experts
- Insight → KM: Store high-confidence insights
- Flip → KM: Store flip events for meta-learning
- Trickster ← KM: Query flip history
"""

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

# Import metrics stubs - will be overwritten if metrics available
try:
    from aragora.server.prometheus_cross_pollination import (
        record_km_inbound_event,
        record_km_outbound_event,
    )
except ImportError:

    def record_km_inbound_event(source: str, event_type: str) -> None:
        pass

    def record_km_outbound_event(target: str, event_type: str) -> None:
        pass


logger = logging.getLogger(__name__)


class KnowledgeMoundHandlersMixin:
    """Mixin providing Knowledge Mound bidirectional event handlers."""

    # Required from parent: _is_km_handler_enabled method
    _is_km_handler_enabled: Callable[[str], bool]

    def _handle_memory_to_mound(self, event: "StreamEvent") -> None:
        """
        Memory stored → Knowledge Mound.

        Sync high-importance memories to Knowledge Mound for cross-debate access.
        Only syncs memories with importance ≥ 0.7 to avoid noise.
        """
        if not self._is_km_handler_enabled("memory_to_mound"):
            return

        data = event.data
        importance = data.get("importance", 0.0)
        content = data.get("content", "")
        tier = data.get("tier", "unknown")

        # Only sync significant memories
        if importance < 0.7:
            return

        logger.debug(
            f"Syncing high-importance memory to KM: importance={importance:.2f}, tier={tier}"
        )

        # Record KM inbound metric
        record_km_inbound_event("memory", event.type.value)

        try:
            from aragora.knowledge.mound import KnowledgeMound
            from aragora.knowledge.mound.adapters.continuum_adapter import (
                ContinuumAdapter,
            )

            # Get or create mound instance
            mound = KnowledgeMound.get_instance()  # type: ignore[attr-defined]
            if mound is None:
                return

            # Use ContinuumAdapter to convert and store
            adapter = ContinuumAdapter()  # type: ignore[call-arg]
            adapter.sync_memory_to_mound(  # type: ignore[call-arg,unused-coroutine]
                content=content,
                importance=importance,
                tier=tier,
                metadata=data.get("metadata", {}),
            )
            logger.info(f"Synced memory to Knowledge Mound (importance={importance:.2f})")

        except ImportError:
            pass  # KnowledgeMound not available
        except Exception as e:
            logger.debug(f"Memory→KM sync failed: {e}")

    def _handle_mound_to_memory_retrieval(self, event: "StreamEvent") -> None:
        """
        Knowledge Mound queried → Memory pre-warm.

        When KM is queried, check for related memories and pre-warm the cache.
        """
        if not self._is_km_handler_enabled("mound_to_memory"):
            return

        data = event.data
        query = data.get("query", "")
        results_count = data.get("results_count", 0)
        workspace_id = data.get("workspace_id", "")

        if not query or results_count == 0:
            return

        logger.debug(f"KM queried, pre-warming memory cache: query='{query[:50]}...'")

        # Record KM outbound metric
        record_km_outbound_event("memory", event.type.value)

        try:
            from aragora.memory import get_continuum_memory

            memory = get_continuum_memory()
            if memory and hasattr(memory, "prewarm_for_query"):
                memory.prewarm_for_query(query, workspace_id=workspace_id)
        except (ImportError, AttributeError):
            pass
        except Exception as e:
            logger.debug(f"KM→Memory pre-warm failed: {e}")

    def _handle_belief_to_mound(self, event: "StreamEvent") -> None:
        """
        Belief network converged → Knowledge Mound.

        Store high-confidence beliefs and cruxes in KM for cross-debate learning.
        """
        if not self._is_km_handler_enabled("belief_to_mound"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        beliefs_count = data.get("beliefs_count", 0)
        cruxes = data.get("cruxes", [])

        logger.debug(f"Belief network converged: {beliefs_count} beliefs, {len(cruxes)} cruxes")

        # Record KM inbound metric
        record_km_inbound_event("belief", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Store converged beliefs
            for belief_data in data.get("beliefs", []):
                if belief_data.get("confidence", 0) >= 0.8:
                    adapter.store_converged_belief(
                        node=belief_data,
                        debate_id=debate_id,
                    )

            # Store cruxes
            for crux_data in cruxes:
                adapter.store_crux(
                    crux=crux_data,
                    debate_id=debate_id,
                    topics=crux_data.get("topics", []),
                )

            logger.info(f"Stored beliefs/cruxes from debate {debate_id}")

        except ImportError:
            pass  # BeliefAdapter not available
        except Exception as e:
            logger.debug(f"Belief→KM storage failed: {e}")

    def _handle_mound_to_belief(self, event: "StreamEvent") -> None:
        """
        Debate start → Initialize belief priors from KM.

        Retrieve historical cruxes and beliefs to initialize priors for new debate.
        """
        if not self._is_km_handler_enabled("mound_to_belief"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        question = data.get("question", "")

        if not question:
            return

        logger.debug(f"Initializing belief priors from KM for debate {debate_id}")

        # Record KM outbound metric
        record_km_outbound_event("belief", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Search for similar historical cruxes
            similar_cruxes = adapter.search_similar_cruxes(
                query=question,
                limit=10,
                min_score=0.3,
            )

            if similar_cruxes:
                logger.info(
                    f"Found {len(similar_cruxes)} historical cruxes relevant to debate {debate_id}"
                )
                # Store in event data for debate to pick up
                # (Actual initialization happens in debate orchestrator)

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Belief initialization failed: {e}")

    def _handle_rlm_to_mound(self, event: "StreamEvent") -> None:
        """
        RLM compression complete → Knowledge Mound.

        Store compression patterns that worked well for future retrieval optimization.
        """
        if not self._is_km_handler_enabled("rlm_to_mound"):
            return

        data = event.data
        compression_ratio = data.get("compression_ratio", 0.0)
        value_score = data.get("value_score", 0.0)
        content_markers = data.get("content_markers", [])

        # Only store high-value compression patterns
        if value_score < 0.7:
            return

        logger.debug(
            f"Storing RLM compression pattern: ratio={compression_ratio:.2f}, value={value_score:.2f}"
        )

        # Record KM inbound metric
        record_km_inbound_event("rlm", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

            adapter = RlmAdapter()
            adapter.store_compression_pattern(
                compression_ratio=compression_ratio,
                value_score=value_score,
                content_markers=content_markers,
                metadata=data.get("metadata", {}),
            )

        except ImportError:
            pass  # RlmAdapter not available yet
        except Exception as e:
            logger.debug(f"RLM→KM storage failed: {e}")

    def _handle_mound_to_rlm(self, event: "StreamEvent") -> None:
        """
        Knowledge Mound queried → RLM priority update.

        Inform RLM about access patterns to optimize compression priorities.
        """
        if not self._is_km_handler_enabled("mound_to_rlm"):
            return

        data = event.data
        query = data.get("query", "")
        results_count = data.get("results_count", 0)
        node_ids = data.get("node_ids", [])

        if not node_ids:
            return

        logger.debug(f"Updating RLM priorities based on KM query: {results_count} results")

        # Record KM outbound metric
        record_km_outbound_event("rlm", event.type.value)

        try:
            from aragora.rlm.compressor import get_compressor  # type: ignore[attr-defined]

            compressor = get_compressor()
            if compressor and hasattr(compressor, "update_priority_hints"):
                compressor.update_priority_hints(
                    accessed_ids=node_ids,
                    query=query,
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→RLM priority update failed: {e}")

    def _handle_elo_to_mound(self, event: "StreamEvent") -> None:
        """
        ELO updated → Knowledge Mound.

        Store agent expertise profiles for cross-debate team selection.
        Only stores significant ELO changes (|delta| > 25).
        """
        if not self._is_km_handler_enabled("elo_to_mound"):
            return

        data = event.data
        agent_name = data.get("agent", "")
        new_elo = data.get("elo", 1500)
        delta = data.get("delta", 0)
        debate_id = data.get("debate_id", "")
        domain = data.get("domain", "general")

        # Only store significant changes
        if abs(delta) < 25:
            return

        logger.debug(
            f"Storing agent expertise: {agent_name} -> {new_elo} (Δ{delta:+.0f}) in {domain}"
        )

        # Record KM inbound metric
        record_km_inbound_event("ranking", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

            adapter = RankingAdapter()
            adapter.store_agent_expertise(
                agent_name=agent_name,
                domain=domain,
                elo=new_elo,
                delta=delta,
                debate_id=debate_id,
            )

        except ImportError:
            pass  # RankingAdapter not available yet
        except Exception as e:
            logger.debug(f"ELO→KM storage failed: {e}")

    def _handle_mound_to_team_selection(self, event: "StreamEvent") -> None:
        """
        Debate start → Query KM for domain experts.

        Retrieve agent expertise profiles to inform team selection.
        """
        if not self._is_km_handler_enabled("mound_to_team"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        question = data.get("question", "")

        if not question:
            return

        logger.debug(f"Querying KM for domain experts for debate {debate_id}")

        # Record KM outbound metric
        record_km_outbound_event("team_selection", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

            adapter = RankingAdapter()
            # Detect domain from question
            domain = adapter.detect_domain(question)
            experts = adapter.get_domain_experts(domain=domain, limit=10)

            if experts:
                logger.info(f"Found {len(experts)} domain experts for '{domain}'")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Team selection query failed: {e}")

    def _handle_insight_to_mound(self, event: "StreamEvent") -> None:
        """
        Insight extracted → Knowledge Mound.

        Store high-confidence insights (≥0.7) for organizational learning.
        """
        if not self._is_km_handler_enabled("insight_to_mound"):
            return

        data = event.data
        confidence = data.get("confidence", 0.0)
        insight_type = data.get("type", "")
        data.get("debate_id", "")

        # Only store high-confidence insights
        if confidence < 0.7:
            return

        logger.debug(f"Storing insight: type={insight_type}, confidence={confidence:.2f}")

        # Record KM inbound metric
        record_km_inbound_event("insights", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

            adapter = InsightsAdapter()
            adapter.store_insight(
                insight=data,  # type: ignore[arg-type]
                min_confidence=0.7,
            )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Insight→KM storage failed: {e}")

    def _handle_flip_to_mound(self, event: "StreamEvent") -> None:
        """
        Flip detected → Knowledge Mound.

        Store ALL flip events for meta-learning and consistency tracking.
        """
        if not self._is_km_handler_enabled("flip_to_mound"):
            return

        data = event.data
        agent_name = data.get("agent_name", "")
        flip_type = data.get("flip_type", "")

        logger.debug(f"Storing flip event: agent={agent_name}, type={flip_type}")

        # Record KM inbound metric
        record_km_inbound_event("trickster", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

            adapter = InsightsAdapter()
            adapter.store_flip(flip=data)  # type: ignore[call-arg,arg-type]

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Flip→KM storage failed: {e}")

    def _handle_mound_to_trickster(self, event: "StreamEvent") -> None:
        """
        Debate start → Query KM for flip history.

        Retrieve agent flip history for consistency prediction.
        """
        if not self._is_km_handler_enabled("mound_to_trickster"):
            return

        data = event.data
        data.get("debate_id", "")
        agents = data.get("agents", [])

        if not agents:
            return

        logger.debug(f"Querying KM for flip history: {len(agents)} agents")

        # Record KM outbound metric
        record_km_outbound_event("trickster", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

            adapter = InsightsAdapter()

            for agent_name in agents:
                flip_history = adapter.get_agent_flip_history(
                    agent_name=agent_name,
                    limit=20,
                )
                if flip_history:
                    logger.debug(f"Found {len(flip_history)} historical flips for {agent_name}")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Trickster query failed: {e}")
