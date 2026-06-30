"""Deprecated re-export shim for the telemetry collector.

The originals now live in :mod:`aragora.observability.telemetry_collector`.
This module re-exports them so the legacy ``aragora.telemetry.collector``
import path keeps working for one release.
"""

import warnings

from aragora.observability.telemetry_collector import (
    InMemoryBackend,
    TelemetryBackend,
    TelemetryCollector,
    get_telemetry_collector,
    record_event,
    set_telemetry_collector,
)

warnings.warn(
    "aragora.telemetry.collector is deprecated; import from "
    "aragora.observability.telemetry_collector instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "InMemoryBackend",
    "TelemetryBackend",
    "TelemetryCollector",
    "get_telemetry_collector",
    "record_event",
    "set_telemetry_collector",
]
