"""Deprecated import location for the distributed tracing middleware.

The tracing middleware moved DOWN to
:mod:`aragora.observability.middleware.tracing` during the P4a layering work so
that foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.middleware.tracing`` still
works but is deprecated; import from
``aragora.observability.middleware.tracing`` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

from aragora.observability.middleware import tracing as _target

warnings.warn(
    "aragora.server.middleware.tracing is deprecated; import from "
    "aragora.observability.middleware.tracing instead.",
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    # Let static type-checkers resolve names through the relocated module.
    def __getattr__(name: str) -> Any: ...
else:
    # Alias the relocated module under the deprecated name so attribute access,
    # from-imports, and test monkeypatches all resolve to the single moved module.
    sys.modules[__name__] = _target
