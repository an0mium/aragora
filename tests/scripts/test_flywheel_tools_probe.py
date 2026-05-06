from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_probe_cli_json_succeeds_when_tools_are_missing(monkeypatch, capsys) -> None:
    import flywheel_tools_probe as mod

    monkeypatch.setattr(mod, "probe_flywheel_tools", lambda **_kwargs: [])

    assert mod.main(["--json", "--no-help"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "flywheel_tools_probe.v1"
    assert payload["mode"] == "read_only_local_probe"
    assert payload["summary"]["tool_count"] == 0
    assert "no tools were installed" in payload["non_claims"]


def test_probe_cli_human_output(monkeypatch, capsys) -> None:
    import flywheel_tools_probe as mod

    monkeypatch.setattr(mod, "probe_flywheel_tools", lambda **_kwargs: [])

    assert mod.main(["--no-help"]) == 0
    output = capsys.readouterr().out

    assert "Flywheel local tool probe" in output
    assert "available: 0 / 0" in output
