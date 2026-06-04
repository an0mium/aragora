"""Tests for ``scripts/runners/generate_runner_health_plist.py``.

Verifies the template renders into a valid, placeholder-free plist that carries
no personal home directory.
"""

from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "runners" / "generate_runner_health_plist.py"
    spec = importlib.util.spec_from_file_location(
        "generate_runner_health_plist_under_test", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_module()


def test_render_substitutes_all_placeholders() -> None:
    out = gen.render("/opt/runner-health/check.sh", "/var/log/aragora")
    private_home = "/Users/" + "armand"
    assert "__RUNNER_HEALTH_SCRIPT__" not in out
    assert "__LOG_DIR__" not in out
    assert private_home not in out


def test_render_injects_paths() -> None:
    out = gen.render("/opt/runner-health/check.sh", "/var/log/aragora")
    parsed = plistlib.loads(out.encode("utf-8"))
    assert parsed["Label"] == "com.aragora.runner-health"
    assert parsed["ProgramArguments"] == ["/opt/runner-health/check.sh"]
    env = parsed["EnvironmentVariables"]
    assert env["LOG_FILE"] == "/var/log/aragora/aragora-runner-health.log"
    assert env["ALERT_FILE"] == "/var/log/aragora/aragora-runner-health.alert"
    assert parsed["StandardOutPath"] == "/var/log/aragora/aragora-runner-health.stdout.log"
    assert parsed["StandardErrorPath"] == "/var/log/aragora/aragora-runner-health.stderr.log"


def test_render_strips_trailing_slash_on_log_dir() -> None:
    out = gen.render("/opt/check.sh", "/var/log/aragora/")
    assert "/var/log/aragora//" not in out
    parsed = plistlib.loads(out.encode("utf-8"))
    assert (
        parsed["EnvironmentVariables"]["LOG_FILE"] == "/var/log/aragora/aragora-runner-health.log"
    )


def test_main_writes_file(tmp_path: Path) -> None:
    out_path = tmp_path / "com.aragora.runner-health.plist"
    rc = gen.main(
        ["--script", "/opt/check.sh", "--log-dir", "/var/log/x", "--output", str(out_path)]
    )
    assert rc == 0
    parsed = plistlib.loads(out_path.read_bytes())
    assert parsed["ProgramArguments"] == ["/opt/check.sh"]


def test_main_honors_env_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_HEALTH_SCRIPT", "/env/check.sh")
    monkeypatch.setenv("RUNNER_HEALTH_LOG_DIR", "/env/logs")
    out_path = tmp_path / "out.plist"
    gen.main(["--output", str(out_path)])
    parsed = plistlib.loads(out_path.read_bytes())
    assert parsed["ProgramArguments"] == ["/env/check.sh"]
    assert parsed["EnvironmentVariables"]["LOG_FILE"] == "/env/logs/aragora-runner-health.log"


def test_default_paths_are_home_rooted() -> None:
    # The committed source derives defaults from Path.home() (no literal user),
    # so at runtime they resolve under the current user's home.
    home = str(Path.home())
    assert str(gen.DEFAULT_SCRIPT).startswith(home)
    assert str(gen.DEFAULT_SCRIPT).endswith("actions-runner/runner-health/mac_timewait_check.sh")
    assert gen.DEFAULT_LOG_DIR == Path.home() / "Library" / "Logs"
