#!/usr/bin/env python3
"""Render the runner-health launchd plist from its template.

Fills machine-specific paths (the health-check script and the log directory) so
the committed artifact carries no personal home directory. Writes to stdout by
default; pass ``--output PATH`` to write a file.

Loading via ``launchctl`` is a deliberate, separate manual step (see
``scripts/runners/README.md``) and is intentionally NOT performed here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("com.aragora.runner-health.plist.template")
DEFAULT_SCRIPT = Path.home() / "actions-runner" / "runner-health" / "mac_timewait_check.sh"
DEFAULT_LOG_DIR = Path.home() / "Library" / "Logs"


def render(script: str, log_dir: str, *, template_path: Path = TEMPLATE_PATH) -> str:
    """Return the plist text with placeholders substituted."""
    text = template_path.read_text(encoding="utf-8")
    rendered = text.replace("__RUNNER_HEALTH_SCRIPT__", script).replace(
        "__LOG_DIR__", log_dir.rstrip("/")
    )
    if "__RUNNER_HEALTH_SCRIPT__" in rendered or "__LOG_DIR__" in rendered:
        raise ValueError("unsubstituted placeholder remains after rendering")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        default=os.environ.get("RUNNER_HEALTH_SCRIPT", str(DEFAULT_SCRIPT)),
        help=(
            "Path to the health-check script "
            "(default: $RUNNER_HEALTH_SCRIPT or ~/actions-runner/runner-health/mac_timewait_check.sh)"
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("RUNNER_HEALTH_LOG_DIR", str(DEFAULT_LOG_DIR)),
        help="Directory for runner-health logs (default: $RUNNER_HEALTH_LOG_DIR or ~/Library/Logs)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output path, or '-' for stdout (default)",
    )
    args = parser.parse_args(argv)

    rendered = render(args.script, args.log_dir)
    if args.output == "-":
        print(rendered, end="")
    else:
        out = Path(args.output).expanduser()
        out.write_text(rendered, encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
