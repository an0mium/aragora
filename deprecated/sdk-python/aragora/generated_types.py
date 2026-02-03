"""
DEPRECATED: Auto-generated types have moved to aragora_sdk.

This file is a backwards compatibility shim. Please update your imports:

    # Old (deprecated):
    from aragora.generated_types import DebateResult, Agent

    # New (recommended):
    from aragora_sdk.generated_types import DebateResult, Agent

This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "aragora.generated_types is deprecated. "
    "Please use aragora_sdk.generated_types instead. "
    "This module will be removed in aragora 3.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location for backwards compatibility
from aragora_sdk.generated_types import *  # noqa: F401, F403, E402
