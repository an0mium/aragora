"""
Operations utilities for Aragora.

DEPRECATED: This package is a backward compatibility shim.
Import from `aragora.ops` instead:

    # New style (preferred)
    from aragora.ops import KeyRotationScheduler, get_key_rotation_scheduler

    # Old style (still works)
    from aragora.operations import KeyRotationScheduler, get_key_rotation_scheduler

Both imports resolve to the same code.
"""

import warnings

warnings.warn(
    "aragora.operations is deprecated. "
    "Import from aragora.ops instead. "
    "This package will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from aragora.ops for backward compatibility
from aragora.ops.key_rotation import (
    KeyRotationScheduler,
    KeyRotationConfig,
    KeyRotationResult,
    get_key_rotation_scheduler,
)

__all__ = [
    "KeyRotationScheduler",
    "KeyRotationConfig",
    "KeyRotationResult",
    "get_key_rotation_scheduler",
]
