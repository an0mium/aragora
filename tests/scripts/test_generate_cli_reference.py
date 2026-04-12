from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_cli_reference.py"


def test_generate_cli_reference_help_smoke() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT.parents[1]),
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate CLI reference documentation" in result.stdout
    assert "--check" in result.stdout
