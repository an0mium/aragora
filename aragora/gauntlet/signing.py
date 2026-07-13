"""Deprecated import location for receipt signing primitives.

Receipt signing moved down to :mod:`aragora.storage.receipt_signing`. This
module retains the gauntlet-facing verification telemetry adapter while
aliasing the relocated module for compatibility.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import TYPE_CHECKING, Any

_target = importlib.import_module("aragora.storage.receipt_signing")


def _emit_receipt_verification(signed_receipt: Any, is_valid: bool) -> None:
    """Adapt storage verification results to the legacy stream event."""
    from aragora.events.types import StreamEvent, StreamEventType
    from aragora.server.stream.emitter import get_global_emitter

    event_name = "RECEIPT_VERIFIED" if is_valid else "RECEIPT_INTEGRITY_FAILED"
    event_type = getattr(StreamEventType, event_name, None)
    if event_type is None:
        return
    emitter = get_global_emitter()
    if emitter is not None:
        emitter.emit(
            StreamEvent(
                type=event_type,
                data={
                    "receipt_id": getattr(signed_receipt, "receipt_id", "unknown"),
                    "valid": is_valid,
                },
            )
        )


_target.set_receipt_verification_observer(_emit_receipt_verification)

warnings.warn(
    "aragora.gauntlet.signing is deprecated; import from aragora.storage.receipt_signing instead.",
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:

    def __getattr__(name: str) -> Any: ...
else:
    # Preserve the default signer and monkeypatch identity at the legacy path.
    sys.modules[__name__] = _target
