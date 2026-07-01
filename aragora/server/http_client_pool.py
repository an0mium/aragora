"""Deprecated import location for the shared HTTP client connection pool.

The HTTP client connection pool moved DOWN to
:mod:`aragora.observability.http_client_pool` during the P4a layering work so
that foundation/infrastructure modules can reach it without importing
``aragora.server``. Importing from ``aragora.server.http_client_pool`` still
works but is deprecated; import from
``aragora.observability.http_client_pool`` instead.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

from aragora.observability import http_client_pool as _target

warnings.warn(
    "aragora.server.http_client_pool is deprecated; import from "
    "aragora.observability.http_client_pool instead.",
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
