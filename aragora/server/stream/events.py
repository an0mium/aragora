"""Deprecated import location for the stream event schemas.

The stream event schema surface (``StreamEvent``, ``StreamEventType``,
``AudienceMessage``) lives in :mod:`aragora.events.types` so that
foundation/infrastructure modules can reach the event dataclasses without
importing ``aragora.server``. Importing them from
``aragora.server.stream.events`` still works but is deprecated; import from
``aragora.events`` (or ``aragora.events.types``) instead.
"""

import warnings

# Re-export from the shared events layer for backward compatibility.
from aragora.events.types import (
    AudienceMessage,
    StreamEvent,
    StreamEventType,
)

warnings.warn(
    "aragora.server.stream.events is deprecated; import StreamEvent, "
    "StreamEventType, and AudienceMessage from aragora.events instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "StreamEventType",
    "StreamEvent",
    "AudienceMessage",
]
