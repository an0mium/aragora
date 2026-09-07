"""Deprecated re-export shim for the simple observer.

The original now lives in :mod:`aragora.observability.simple_observer`.
This module re-exports it so the legacy ``aragora.monitoring.simple_observer``
import path keeps working for one release.
"""

import warnings

from aragora.observability.simple_observer import SimpleObserver

warnings.warn(
    "aragora.monitoring.simple_observer is deprecated; import from "
    "aragora.observability.simple_observer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SimpleObserver"]
