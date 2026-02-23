"""
Gauntlet shim for backward compatibility.

DEPRECATED: This module is a backward compatibility shim.
Import from ``aragora.gauntlet`` instead.

    # New style (preferred)
    from aragora.gauntlet import GauntletRunner, GauntletConfig

    # Old style (still works but will be removed)
    from aragora.modes.gauntlet import GauntletRunner
"""

import warnings

warnings.warn(
    "aragora.modes.gauntlet is deprecated. "
    "Import from aragora.gauntlet instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location for backward compatibility
from aragora.gauntlet import (  # noqa: E402
    AttackCategory,
    GauntletConfig,
    GauntletResult,
    GauntletRunner,
    ProbeCategory,
    Vulnerability,
)

__all__ = [
    "GauntletRunner",
    "GauntletConfig",
    "GauntletResult",
    "AttackCategory",
    "ProbeCategory",
    "Vulnerability",
]
