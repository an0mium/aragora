#!/usr/bin/env python3
"""Compatibility entrypoint for refreshing the automation status cache."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.cache_codex_automation_github_status as cache_status


def main(argv: list[str] | None = None) -> int:
    return cache_status.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
