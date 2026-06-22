#!/usr/bin/env python3
"""Compatibility entrypoint for ``reconcile_automation_outbox.py``."""

from __future__ import annotations

from reconcile_automation_outbox import main


if __name__ == "__main__":
    raise SystemExit(main())
