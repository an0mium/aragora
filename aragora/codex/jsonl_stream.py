"""Streaming JSONL reader for Codex Desktop rollout files.

Rollout files can be tens or hundreds of MB. Callers must never slurp them
whole — iterate line by line via :func:`iter_jsonl`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | PathLike[str], *, strict: bool = True) -> Iterator[dict[str, Any]]:
    """Yield one parsed dict per non-blank line of ``path``.

    Blank lines are skipped silently. In strict mode, malformed JSON lines
    raise :class:`json.JSONDecodeError` with the file path and line number
    attached in the message so failures are debuggable without re-reading the
    file. In non-strict mode, iteration stops at the first malformed line; this
    is useful for live Codex rollout files that may end with a partially written
    JSON object.
    """
    abs_path = Path(path).expanduser()
    with abs_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if not strict:
                    break
                raise json.JSONDecodeError(
                    f"{exc.msg} ({abs_path}:{lineno})",
                    exc.doc,
                    exc.pos,
                ) from None
            if isinstance(obj, dict):
                yield obj
            # Silently skip non-object lines — rollouts may contain stray
            # arrays or scalars from legacy event schemas, and the inspector
            # is intentionally tolerant of unknown shapes.
