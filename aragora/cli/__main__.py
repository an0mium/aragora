"""Package entry point for ``python -m aragora.cli``."""

from __future__ import annotations

from aragora.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
