"""Focused tests for the fail-closed baseline-aware mypy gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "mypy_with_baseline.py"


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mypy_with_baseline_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_validate_toolchain_versions", lambda: None)
    return module


def _result(returncode: int, output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=output)


def test_known_baseline_diagnostics_can_pass(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = "aragora/example.py:12: error: existing debt [assignment]\n"
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(1, output))
    observed: dict[str, str] = {}

    def fake_filter(value: str) -> int:
        observed["output"] = value
        return 0

    monkeypatch.setattr(gate, "_filter", fake_filter)

    assert gate.main([]) == 0
    assert observed == {"output": output}


def test_new_diagnostics_propagate_baseline_failure(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = "aragora/example.py:12:4: error: new debt [assignment]\n"
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(1, output))
    monkeypatch.setattr(gate, "_filter", lambda value: 1)

    assert gate.main([]) == 1


def test_clean_mypy_output_reaches_baseline_filter(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(0, "Success\n"))
    observed: dict[str, str] = {}

    def fake_filter(value: str) -> int:
        observed["output"] = value
        return 0

    monkeypatch.setattr(gate, "_filter", fake_filter)

    assert gate.main([]) == 0
    assert observed == {"output": "Success\n"}


def test_default_scan_is_non_incremental(gate: ModuleType) -> None:
    assert "--no-incremental" in gate.DEFAULT_MYPY_ARGS


def test_missing_mypy_module_fails_closed(
    gate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = "/usr/bin/python3: No module named mypy\n"
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(1, output))
    monkeypatch.setattr(
        gate,
        "_filter",
        lambda value: pytest.fail("tool failure must not reach the baseline filter"),
    )

    assert gate.main([]) == gate.TOOL_FAILURE
    assert output.strip() in capsys.readouterr().err


def test_process_start_failure_fails_closed(
    gate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_start(args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        raise OSError("exec failed")

    monkeypatch.setattr(gate, "_run_mypy", fail_to_start)

    assert gate.main([]) == gate.TOOL_FAILURE
    assert "could not start mypy: exec failed" in capsys.readouterr().err


def test_toolchain_version_mismatch_fails_closed_before_mypy(
    gate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gate,
        "_validate_toolchain_versions",
        lambda: "required mypy==2.3.0, found 2.1.0",
    )
    monkeypatch.setattr(
        gate,
        "_run_mypy",
        lambda args: pytest.fail("mypy must not run with the wrong toolchain"),
    )

    assert gate.main([]) == gate.TOOL_FAILURE
    assert "required mypy==2.3.0, found 2.1.0" in capsys.readouterr().err


def test_unexpected_mypy_status_fails_closed(
    gate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(2, "config error\n"))

    assert gate.main([]) == gate.TOOL_FAILURE
    captured = capsys.readouterr().err
    assert "config error" in captured
    assert "unexpected status 2" in captured


def test_exit_one_without_diagnostics_fails_closed(
    gate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(1, "opaque failure\n"))

    assert gate.main([]) == gate.TOOL_FAILURE
    captured = capsys.readouterr().err
    assert "opaque failure" in captured
    assert "without recognized" in captured


def test_sync_uses_validated_mypy_output(gate: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    output = "aragora/example.py:12: error: existing debt [assignment]\n"
    monkeypatch.setattr(gate, "_run_mypy", lambda args: _result(1, output))
    observed: dict[str, str] = {}

    def fake_sync(value: str) -> int:
        observed["output"] = value
        return 0

    monkeypatch.setattr(gate, "_sync", fake_sync)

    assert gate.main(["--sync"]) == 0
    assert observed == {"output": output}
