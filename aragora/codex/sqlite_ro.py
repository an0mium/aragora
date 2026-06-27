"""Read-only SQLite access for the Codex Desktop state database.

Opens the underlying SQLite file with the ``mode=ro`` URI so the connection
cannot acquire a write lock. The live Codex Desktop app holds a separate write
handle on this file (WAL mode is in use); the read-only mode here is the
discipline that keeps the inspector from interfering with it.
"""

from __future__ import annotations

import sqlite3
from urllib.parse import quote
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path


@contextmanager
def sqlite_ro(path: str | PathLike[str]) -> Iterator[sqlite3.Connection]:
    """Yield a read-only :class:`sqlite3.Connection` for ``path``.

    The connection is closed on context exit. ``row_factory`` is set to
    :class:`sqlite3.Row` so callers can use column-name access. Attempting any
    write through the returned connection raises
    :class:`sqlite3.OperationalError` from SQLite itself.
    """
    abs_path = Path(path).expanduser().resolve()
    uri = f"file:{quote(str(abs_path), safe='/:')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()
