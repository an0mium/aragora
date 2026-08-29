#!/usr/bin/env python3
"""Run Aragora migrations after loading DATABASE_URL from mounted custody."""

from __future__ import annotations

import os
import sys

from aragora.config.secrets import hydrate_env_from_secrets
from aragora.migrations.runner import apply_migrations


def main() -> int:
    hydrated = hydrate_env_from_secrets(["DATABASE_URL"], overwrite=True)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url or "DATABASE_URL" not in hydrated:
        raise RuntimeError("DATABASE_URL was not loaded from managed custody")
    applied = apply_migrations(database_url=database_url)
    sys.stdout.write(f"Applied {len(applied)} migration(s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
