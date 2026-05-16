"""Tiny duration-string parser for the codex inspector CLI.

Accepts ``<int><unit>`` where unit is ``s``, ``m``, ``h``, or ``d``. Centralizing
this here (rather than depending on a heavier package) keeps the codex inspector
free of additional imports.
"""

from __future__ import annotations

import re
from datetime import timedelta

_PATTERN = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")
_UNIT_TO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str) -> timedelta:
    """Parse ``"4h" | "30m" | "1d" | "90s"`` into a :class:`timedelta`.

    Raises ``ValueError`` with a clear message on bad input. Zero durations are
    allowed (``"0s"``); negative durations are not (the regex rejects ``-``).
    """
    match = _PATTERN.match(value or "")
    if match is None:
        raise ValueError(
            f"invalid duration {value!r}: expected '<int><unit>' with unit in s|m|h|d "
            "(e.g. '4h', '30m', '1d', '90s')"
        )
    amount = int(match.group(1))
    unit = match.group(2)
    return timedelta(seconds=amount * _UNIT_TO_SECONDS[unit])
