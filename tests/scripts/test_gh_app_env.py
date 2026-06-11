"""Tests for ``scripts/gh_app_env.py`` (per-pass GitHub App token for shell ``gh``).

Covers the three contract-critical behaviors:

* **Config-absent silence** — when no App config is available the script must
  exit 0 and print *nothing* (daemons degrade to the operator's existing gh
  auth instead of crashing).
* **Token print path** — ``--print-token`` emits exactly the bare token on
  stdout for command substitution; the default mode emits eval-able
  ``GH_TOKEN=...`` shell assignments.
* **No leakage** — the token never appears on stderr, in diagnostics, or in
  the config-absent path.

The minter boundary is injected; no test touches the network. One subprocess
integration test exercises the real script end-to-end with App auth disabled.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gh_app_env.py"

FAKE_TOKEN = "ghs_test_token_abc123"  # noqa: S105 - synthetic test fixture


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("gh_app_env_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load_module()


# ---------------------------------------------------------------------------
# Config-absent: silent-safe degradation
# ---------------------------------------------------------------------------


def test_config_absent_exits_zero_and_prints_nothing(mod, capsys) -> None:
    rc = mod.main(["--print-token"], minter=lambda: None)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""


def test_config_absent_quiet_emits_no_stderr(mod, capsys) -> None:
    rc = mod.main(["--print-token", "--quiet"], minter=lambda: None)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


def test_config_absent_env_mode_prints_nothing(mod, capsys) -> None:
    rc = mod.main([], minter=lambda: None)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""


def test_minter_crash_is_silent_safe(mod, capsys) -> None:
    def _boom() -> str:
        raise RuntimeError("mint exploded")

    rc = mod.main(["--print-token", "--quiet"], minter=_boom)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


def test_import_failure_is_silent_safe(mod, capsys, monkeypatch) -> None:
    def _import_error() -> Any:
        raise ImportError("aragora not importable")

    monkeypatch.setattr(mod, "_resolve_minter", _import_error)
    rc = mod.main(["--print-token", "--quiet"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# Token print paths
# ---------------------------------------------------------------------------


def test_print_token_emits_bare_token_only(mod, capsys) -> None:
    rc = mod.main(["--print-token"], minter=lambda: FAKE_TOKEN)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == FAKE_TOKEN + "\n"
    assert FAKE_TOKEN not in captured.err


def test_default_mode_emits_eval_able_assignment(mod, capsys) -> None:
    rc = mod.main([], minter=lambda: FAKE_TOKEN)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == f"GH_TOKEN={FAKE_TOKEN}\n"
    assert FAKE_TOKEN not in captured.err


def test_no_token_leakage_to_stderr_in_any_mode(mod, capsys) -> None:
    for argv in (["--print-token"], [], ["--print-token", "--quiet"], ["--quiet"]):
        mod.main(argv, minter=lambda: FAKE_TOKEN)
        captured = capsys.readouterr()
        assert FAKE_TOKEN not in captured.err


def test_whitespace_token_treated_as_absent(mod, capsys) -> None:
    rc = mod.main(["--print-token", "--quiet"], minter=lambda: "   ")
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Subprocess integration: real script, App auth disabled
# ---------------------------------------------------------------------------


def test_subprocess_config_absent_silent(tmp_path) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GITHUB_APP", "GH_APP", "ARAGORA_GITHUB"))
    }
    env["ARAGORA_DISABLE_GITHUB_APP_TOKEN"] = "1"
    env["ARAGORA_AUTOMATION_ENV_FILE"] = str(tmp_path / "missing.env")
    result = subprocess.run(  # noqa: S603 - test invokes our own script
        [sys.executable, str(SCRIPT_PATH), "--print-token", "--quiet"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert result.stdout == ""
    # stderr may carry unrelated import-time warnings from the aragora package
    # (daemons discard stderr); the contract is: no script diagnostic under
    # --quiet and no token-shaped material ever.
    assert "gh_app_env:" not in result.stderr
    assert "ghs_" not in result.stderr
