from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_openapi_generated_artifacts_are_current(tmp_path: Path) -> None:
    generated = {
        name: tmp_path / name for name in ("openapi_generated.json", "openapi_generated.yaml")
    }
    for name, fmt in (("openapi_generated.json", "json"), ("openapi_generated.yaml", "yaml")):
        subprocess.run(
            [
                sys.executable,
                "scripts/generate_openapi.py",
                "--output",
                str(generated[name]),
                "--format",
                fmt,
            ],
            cwd=ROOT,
            check=True,
        )
    for script in (
        "add_openapi_operation_ids.py",
        "add_openapi_param_descriptions.py",
        "add_openapi_descriptions.py",
    ):
        subprocess.run(
            [
                sys.executable,
                f"scripts/{script}",
                "--spec",
                str(generated["openapi_generated.json"]),
            ],
            cwd=ROOT,
            check=True,
        )
    for name, output in generated.items():
        committed = ROOT / "docs" / "api" / name
        if output.read_text() != committed.read_text():
            diff = "".join(
                difflib.unified_diff(
                    committed.read_text().splitlines(True),
                    output.read_text().splitlines(True),
                    fromfile=str(committed),
                    tofile=str(output),
                )
            )
            pytest.fail(f"{name} is stale.\n{diff[:4000]}")
