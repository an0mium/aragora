#!/usr/bin/env python3
"""Compatibility entrypoint for reading operator-steering mailboxes.

The implementation lives in ``read_operator_steering.py``.  This wrapper
provides the user-facing verb used by session startup hooks and operator
prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import read_operator_steering


def main(argv: list[str] | None = None) -> int:
    return read_operator_steering.main(argv, prog=Path(__file__).name)


if __name__ == "__main__":
    sys.exit(main())
