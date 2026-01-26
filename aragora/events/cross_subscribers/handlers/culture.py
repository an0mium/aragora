"""
Culture-related event handlers.

Handles organizational culture patterns and debate protocol hints:
- Culture patterns → Debate protocol
- Debate start → Load culture from KM
- Knowledge staleness → Debate warnings
"""

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

# Import metrics stubs - will be overwritten if metrics available
try:
    from aragora.server.prometheus_cross_pollination import record_km_outbound_event
except ImportError:

    def record_km_outbound_event(target: str, event_type: str) -> None:
        pass


logger = logging.getLogger(__name__)


class CultureHandlersMixin:
    """Mixin providing culture-related event handlers."""

    # Required from parent: _is_km_handler_enabled method
    _is_km_handler_enabled: Callable[[str], bool]

    # Culture storage dict - will be initialized lazily
    _debate_cultures: dict

    def _handle_culture_to_debate(self, event: "StreamEvent") -> None:
        """
        Culture patterns updated → Debate protocol.

        When culture patterns emerge, inform debate protocol selection.
        Only handles MOUND_UPDATED events with type=culture_patterns.
        """
        if not self._is_km_handler_enabled("culture_to_debate"):
            return

        data = event.data
        update_type = data.get("update_type", "")

        if update_type != "culture_patterns":
            return

        patterns_count = data.get("patterns_count", 0)
        workspace_id = data.get("workspace_id", "")

        logger.debug(
            f"Culture patterns available: {patterns_count} patterns in workspace {workspace_id}"
        )

        # Culture patterns are used passively during debate initialization
        # by querying the CultureAccumulator

    def _handle_mound_to_culture(self, event: "StreamEvent") -> None:
        """
        Debate start → Load culture patterns from KM.

        Retrieve relevant culture patterns when a debate starts to inform
        protocol selection and agent behavior. Patterns include:
        - Decision style preferences (consensus vs majority)
        - Risk tolerance (conservative vs aggressive)
        - Domain expertise distribution
        - Debate dynamics (rounds to consensus, critique patterns)
        """
        if not self._is_km_handler_enabled("mound_to_culture"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        domain = data.get("domain", "")
        data.get("protocol", {})

        logger.debug(f"Loading culture patterns for debate {debate_id}, domain={domain}")

        # Record KM outbound metric
        record_km_outbound_event("culture", event.type.value)

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if not mound:
                logger.debug("Knowledge Mound not available for culture retrieval")
                return

            # Check if mound is initialized
            if not mound.is_initialized:
                logger.debug("Knowledge Mound not initialized, skipping culture retrieval")
                return

            # Retrieve culture profile from mound
            import asyncio

            async def retrieve_culture():
                if hasattr(mound, "get_culture_profile"):
                    profile = await mound.get_culture_profile()
                    return profile
                return None

            # Run async retrieval
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task for later execution
                    asyncio.create_task(retrieve_culture())
                else:
                    profile = loop.run_until_complete(retrieve_culture())
                    if profile:
                        self._store_debate_culture(debate_id, profile, domain)
            except RuntimeError:
                profile = asyncio.run(retrieve_culture())
                if profile:
                    self._store_debate_culture(debate_id, profile, domain)

        except ImportError as e:
            logger.debug(f"Culture retrieval import failed: {e}")
        except Exception as e:
            logger.debug(f"Culture→Debate retrieval failed: {e}")

    def _store_debate_culture(
        self,
        debate_id: str,
        profile: Any,
        domain: str,
    ) -> None:
        """Store culture profile for a debate to inform protocol behavior.

        Args:
            debate_id: Debate identifier
            profile: CultureProfile from Knowledge Mound
            domain: Detected debate domain
        """
        try:
            # Store culture context for this debate
            # This can be accessed by the orchestrator during debate execution
            if not hasattr(self, "_debate_cultures"):
                self._debate_cultures = {}

            # Extract relevant protocol hints from culture
            protocol_hints = {}

            if hasattr(profile, "dominant_pattern"):
                dominant = profile.dominant_pattern
                if dominant:
                    # Map decision style to protocol recommendations
                    if hasattr(dominant, "pattern_type"):
                        if str(dominant.pattern_type) == "decision_style":
                            protocol_hints["recommended_consensus"] = dominant.value

                    # Map risk tolerance to critique depth
                    if hasattr(dominant, "pattern_type"):
                        if str(dominant.pattern_type) == "risk_tolerance":
                            if dominant.value == "conservative":
                                protocol_hints["extra_critique_rounds"] = 1
                            elif dominant.value == "aggressive":
                                protocol_hints["early_consensus_threshold"] = 0.7

            # Extract domain-specific patterns
            if hasattr(profile, "patterns"):
                domain_patterns = [
                    p for p in profile.patterns if hasattr(p, "domain") and p.domain == domain
                ]
                if domain_patterns:
                    protocol_hints["domain_patterns"] = [
                        {
                            "type": str(p.pattern_type),
                            "value": p.value,
                            "confidence": p.confidence,
                        }
                        for p in domain_patterns
                    ]

            self._debate_cultures[debate_id] = {
                "profile": profile,
                "protocol_hints": protocol_hints,
                "domain": domain,
            }

            logger.info(
                f"Stored culture context for debate {debate_id}: {len(protocol_hints)} hints"
            )

        except Exception as e:
            logger.debug(f"Failed to store debate culture: {e}")

    def get_debate_culture_hints(self, debate_id: str) -> dict:
        """Get protocol hints from culture for a debate.

        Args:
            debate_id: Debate identifier

        Returns:
            Dict of protocol hints derived from organizational culture
        """
        if not hasattr(self, "_debate_cultures"):
            return {}

        culture_ctx = self._debate_cultures.get(debate_id, {})
        return culture_ctx.get("protocol_hints", {})

    def _handle_staleness_to_debate(self, event: "StreamEvent") -> None:
        """
        Knowledge stale → Debate warning.

        When knowledge becomes stale, check if any active debate cites it.
        """
        if not self._is_km_handler_enabled("staleness_to_debate"):
            return

        data = event.data
        node_id = data.get("node_id", "")
        staleness_reason = data.get("reason", "")
        data.get("last_verified", "")

        logger.debug(f"Knowledge stale: {node_id} - {staleness_reason}")

        # Record KM outbound metric (staleness warning to debate)
        record_km_outbound_event("debate", event.type.value)

        try:
            from aragora.server.stream.state_manager import get_active_debates

            active_debates = get_active_debates()

            # Check if any active debate references this node
            for debate_id, debate_state in active_debates.items():
                cited_nodes = debate_state.get("cited_knowledge", [])
                if node_id in cited_nodes:
                    logger.warning(f"Active debate {debate_id} cites stale knowledge: {node_id}")
                    # Could emit a warning event to the debate here

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Staleness→Debate check failed: {e}")
