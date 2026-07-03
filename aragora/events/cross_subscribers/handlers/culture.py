"""
Culture-related event handlers.

Handles organizational culture patterns and debate protocol hints:
- Culture patterns → Debate protocol

Debate start → Load culture from KM (``mound_to_culture``) relocated to
``aragora.knowledge.event_subscribers.KnowledgeEventSubscriber`` (P4a Batch E2c);
it is knowledge-coupled while this module is not. Knowledge staleness → Debate
warning (``staleness_to_debate``) relocated to
``aragora.server.event_subscribers.ServerEventSubscriber`` (P4a Batch E6
relocate-UP); it is server-coupled (``server.stream.state_manager``) while this
module is not.
"""

import logging
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

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
