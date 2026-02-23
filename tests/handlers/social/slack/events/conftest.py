"""Conftest for Slack events handler tests.

Sets up sys.modules alias so the relative import
``from .._slack_impl import ...`` in handlers.py resolves correctly.
"""

from __future__ import annotations

import importlib
import sys

# The handler uses ``from .._slack_impl import SLACK_BOT_TOKEN, create_tracked_task``
# which resolves to ``aragora.server.handlers.social.slack._slack_impl``.
# The real module lives at ``aragora.server.handlers.social._slack_impl``.
# Create an alias so the import works.

_ALIAS_KEY = "aragora.server.handlers.social.slack._slack_impl"
_REAL_KEY = "aragora.server.handlers.social._slack_impl"

if _REAL_KEY not in sys.modules:
    try:
        importlib.import_module(_REAL_KEY)
    except ImportError:
        pass

if _REAL_KEY in sys.modules:
    sys.modules[_ALIAS_KEY] = sys.modules[_REAL_KEY]
