"""Tests for scripts/run_test_baseline.py."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_test_baseline.py"
    spec = importlib.util.spec_from_file_location("run_test_baseline", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load run_test_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_pytest_command_includes_default_ignores_and_collect_only() -> None:
    module = _load_script_module()
    args = argparse.Namespace(
        paths=["tests/"],
        markers=module.DEFAULT_MARKERS,
        timeout=120,
        maxfail=1,
        run=False,
        verbose=False,
    )

    command = module._build_pytest_command(args)

    for ignored_path in module.DEFAULT_IGNORE_PATHS:
        assert ["--ignore", ignored_path] in [
            command[index : index + 2] for index in range(len(command) - 1)
        ]
    assert "--collect-only" in command
    assert "-q" in command
