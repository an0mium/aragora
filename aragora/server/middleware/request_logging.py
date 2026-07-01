"""Deprecated import location for the request logging middleware.

The request logging middleware moved DOWN to
:mod:`aragora.observability.middleware.request_logging` during the P4a layering
work so that foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.middleware.request_logging``
still works but is deprecated; import from
``aragora.observability.middleware.request_logging`` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

from aragora.observability.middleware import request_logging as _target

warnings.warn(
    "aragora.server.middleware.request_logging is deprecated; import from "
    "aragora.observability.middleware.request_logging instead.",
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
