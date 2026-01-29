"""
OAuth utility functions.

Provides the ``_impl()`` helper that all mixin modules use to resolve config
functions and utility names from the ``_oauth_impl`` backward-compatibility
shim at runtime.  This ensures that ``unittest.mock.patch`` calls targeting
``_oauth_impl.<name>`` are visible to the actual mixin code.

Also holds the shared ``_oauth_limiter`` instance.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType

from aragora.server.handlers.utils.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

# Rate limiter for OAuth endpoints (20 requests per minute - auth attempts should be limited)
_oauth_limiter = RateLimiter(requests_per_minute=20)

# Module path constant for the backward-compat shim that tests patch against.
_IMPL_MODULE = "aragora.server.handlers._oauth_impl"


def _impl() -> ModuleType:
    """Return the ``_oauth_impl`` module lazily to avoid circular imports.

    All mixin modules call ``_impl().<name>`` instead of importing config
    functions directly so that ``unittest.mock.patch`` applied to
    ``_oauth_impl.<name>`` is visible to the running code.
    """
    return sys.modules[_IMPL_MODULE]
