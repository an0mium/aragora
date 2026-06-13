from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_cross_sdk_parity.py"


def _load_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "check_cross_sdk_parity_under_test",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _BrokenStdout:
    def close(self) -> None:
        return None

    def flush(self) -> None:
        raise BrokenPipeError("downstream closed")

    def write(self, _text: str) -> int:
        raise BrokenPipeError("downstream closed")


def test_json_output_reports_parity_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_extract_python_paths", lambda: {"/v1/a", "/v1/b"})
    monkeypatch.setattr(mod, "_extract_typescript_paths", lambda: {"/v1/b", "/v1/c"})

    assert mod.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["python_endpoint_count"] == 2
    assert payload["typescript_endpoint_count"] == 2
    assert payload["common_count"] == 1
    assert payload["python_only"] == ["/v1/a"]
    assert payload["typescript_only"] == ["/v1/c"]


def test_json_output_suppresses_downstream_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_extract_python_paths", lambda: {"/v1/a"})
    monkeypatch.setattr(mod, "_extract_typescript_paths", lambda: {"/v1/a"})
    monkeypatch.setattr(sys, "stdout", _BrokenStdout())

    assert mod.main(["--json"]) == 0
    redirected = sys.stdout
    try:
        assert getattr(redirected, "name", None) == os.devnull
    finally:
        redirected.close()
