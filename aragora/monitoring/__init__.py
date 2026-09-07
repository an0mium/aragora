"""Deprecated monitoring package.

System-health monitoring has moved into :mod:`aragora.observability`.
``SimpleObserver`` now lives at :mod:`aragora.observability.simple_observer`;
this package re-exports it so the legacy ``aragora.monitoring`` import path
keeps working for one release.
"""

import warnings

from aragora.observability.simple_observer import SimpleObserver

warnings.warn(
    "aragora.monitoring is deprecated; import from aragora.observability instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SimpleObserver"]
