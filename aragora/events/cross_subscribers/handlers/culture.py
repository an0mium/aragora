"""
Culture-related event handlers.

Handles organizational culture patterns and debate protocol hints:
- Culture patterns → Debate protocol
- Knowledge staleness → Debate warnings

Debate start → Load culture from KM (``mound_to_culture``) relocated to
``aragora.knowledge.event_subscribers.KnowledgeEventSubscriber`` (P4a Batch E2c);
it is knowledge-coupled while this module is not.
"""

import logging
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

# Import metrics stubs - will be overwritten if metrics available
try:
    from aragora.observability.prometheus_cross_pollination import record_km_outbound_event
except ImportError:

    def record_km_outbound_event(target: str, event_type: str) -> None:
        pass


logger = logging.getLogger(__name__)


class CultureHandlersMixin:
    """Mixin providing culture-related event handlers."""

    # Required from parent: _is_km_handler_enabled method
    _is_km_handler_enabled: Callable[[str], bool]

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

        logger.debug("Knowledge stale: %s - %s", node_id, staleness_reason)

        # Record KM outbound metric (staleness warning to debate)
        record_km_outbound_event("debate", event.type.value)

        try:
            from aragora.server.stream.state_manager import get_active_debates

            active_debates = get_active_debates()

            # Check if any active debate references this node
            for debate_id, debate_state in active_debates.items():
                cited_nodes = debate_state.get("cited_knowledge", [])
                if node_id in cited_nodes:
                    logger.warning("Active debate %s cites stale knowledge: %s", debate_id, node_id)
                    # Could emit a warning event to the debate here

        except ImportError:
            pass
        except (RuntimeError, TypeError, AttributeError, ValueError, KeyError) as e:
            logger.debug("Staleness→Debate check failed: %s", e)
