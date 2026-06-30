"""Deprecated import location for the SQLite-backed debate storage surface.

The debate-storage surface moved down to :mod:`aragora.storage.debate_storage`
during the P4a layering work so that infrastructure-layer modules (e.g.
``aragora.storage.adapters``) can reach it without importing ``aragora.server``.
Importing from ``aragora.server.storage`` still works but is deprecated; import
from ``aragora.storage.debate_storage`` instead.
"""

from __future__ import annotations

import warnings

from aragora.storage.debate_storage import (
    DB_TIMEOUT,
    DebateMetadata,
    DebateStorage,
    _escape_like_pattern,
    _validate_sql_identifier,
    get_debates_db,
)

warnings.warn(
    "aragora.server.storage is deprecated; import from aragora.storage.debate_storage instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DebateStorage",
    "DebateMetadata",
    "get_debates_db",
    "DB_TIMEOUT",
    "_escape_like_pattern",
    "_validate_sql_identifier",
]
