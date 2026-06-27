#!/usr/bin/env python3
"""Compatibility entrypoint for ``publisher_freshness_check.py``."""

from __future__ import annotations

from publisher_freshness_check import main


if __name__ == "__main__":
    raise SystemExit(main())
