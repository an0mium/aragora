"""Unit tests for scripts/ci/check_tool_baseline.py and tool_baseline_parsers.py.

Parsers are exercised against captured real tool output under
``tests/ci/fixtures/tool_baseline/`` (stdout saved verbatim). The runner is
exercised in-process through ``main(argv)`` with a Python "fake tool" that
replays a fixture and exits with a chosen code, so the suite never depends on
ruff/mypy/grep being on PATH. The end-to-end flow against the real tools is
the VAL-RATCHET-002..016 acceptance checks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
FIXTURES = REPO_ROOT / "tests" / "ci" / "fixtures" / "tool_baseline"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_CI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parsers = _load("tool_baseline_parsers")
ctb = _load("check_tool_baseline")

# Source files the fixtures were captured from (ruff/mypy keys hash these lines).
_SOURCE_FILES = {
    "pkg/__init__.py": "",
    "pkg/mod.py": (
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def bad() -> str:\n"
        "    # TODO: make this real\n"
        '    return add(1, "2")\n'
        "\n"
        "\n"
        'x: int = "not an int"  # FIXME later\n'
    ),
    "other.py": (
        "from typing import List  # TODO remove\n"
        "\n"
        "\n"
        "def f(xs: List[int]) -> None:\n"
        "    unused = 1\n"
    ),
}


def _write_sources(root: Path) -> None:
    for rel, text in _SOURCE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _fake_tool(root: Path, stdout: str, rc: int = 0, stderr: str = "") -> list[str]:
    """A command that prints ``stdout`` verbatim and exits ``rc``."""
    script = root / f"fake_tool_{abs(hash((stdout, rc, stderr)))}.py"
    script.write_text(
        f"import sys\nsys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\nsys.exit({rc})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _run(
    tool: str,
    baseline: Path,
    cwd: Path,
    command: list[str],
    *extra: str,
) -> int:
    argv = ["--tool", tool, "--baseline", str(baseline), "--cwd", str(cwd), *extra, "--", *command]
    return ctb.main(argv)


@pytest.fixture
def fx(tmp_path: Path) -> Path:
    """Source tree matching the captured fixtures, with the fake tools alongside."""
    _write_sources(tmp_path)
    return tmp_path


RUFF_OUT = (FIXTURES / "ruff-concise.txt").read_text(encoding="utf-8")
MYPY_OUT = (FIXTURES / "mypy.txt").read_text(encoding="utf-8")
TODO_OUT = (FIXTURES / "todo-grep.txt").read_text(encoding="utf-8")


# --- parsers ----------------------------------------------------------------


def test_parse_ruff_concise_fixture():
    findings = parsers.parse_ruff(RUFF_OUT)
    assert [(f.path, f.line, f.rule) for f in findings] == [
        ("other.py", 5, "F841"),
        ("pkg/mod.py", 1, "F401"),
        ("pkg/mod.py", 2, "F401"),
    ]
    # The "Found 3 errors." / fixable trailer lines are not findings.
    assert all(f.symbol == "" for f in findings)  # runner hashes the source line
    assert parsers.PARSERS["ruff"].symbol_from_line is True


def test_parse_ruff_handles_syntax_error_lines():
    out = "a.py:3:1: SyntaxError: unexpected indentation\n"
    (finding,) = parsers.parse_ruff(out)
    assert (finding.path, finding.line, finding.rule) == ("a.py", 3, "SyntaxError")


def test_parse_mypy_fixture_skips_notes_and_summary():
    findings = parsers.parse_mypy(MYPY_OUT)
    assert [(f.path, f.line, f.rule) for f in findings] == [
        ("pkg/mod.py", 11, "return-value"),
        ("pkg/mod.py", 11, "arg-type"),
        ("pkg/mod.py", 14, "assignment"),
    ]
    notes = "a.py:1: note: See https://mypy.rtfd.io\nFound 1 error in 1 file\n"
    assert parsers.parse_mypy(notes) == []


def test_parse_mypy_column_numbers_and_missing_code():
    out = "a.py:4:9: error: Something broke\n"
    (finding,) = parsers.parse_mypy(out)
    assert (finding.path, finding.line, finding.rule) == ("a.py", 4, "error")


def test_parse_todo_grep_fixture_hashes_content_and_uses_marker_as_rule():
    findings = parsers.parse_todo(TODO_OUT)
    assert [(f.path, f.line, f.rule) for f in findings] == [
        ("./other.py", 1, "TODO"),
        ("./pkg/mod.py", 10, "TODO"),
        ("./pkg/mod.py", 14, "FIXME"),
    ]
    assert findings[1].symbol == parsers.line_hash("    # TODO: make this real")
    assert len(findings[1].symbol) == parsers.LINE_HASH_LENGTH
    # grep exit 1 (no matches) is a clean run for the todo parser.
    assert parsers.PARSERS["todo"].clean_exit_codes == frozenset({0, 1})


def test_every_registered_parser_returns_empty_on_empty_stdout():
    for name, spec in parsers.PARSERS.items():
        assert spec.parse("") == [], name
        assert spec.description and spec.example_command, name


def test_registry_makes_adding_a_parser_a_one_function_change(monkeypatch):
    monkeypatch.setattr(parsers, "PARSERS", dict(parsers.PARSERS))

    @parsers.register("faketool", description="d", example_command="faketool .")
    def parse_faketool(stdout: str) -> list:
        return [parsers.Finding(path="a.py", rule="X1", symbol="sym")]

    assert "faketool" in parsers.supported_tools()
    assert parsers.PARSERS["faketool"].parse("anything")[0].key() == "a.py::sym::X1"
    with pytest.raises(ValueError):
        parsers.register("faketool", description="d", example_command="x")(parse_faketool)


# --- keys -------------------------------------------------------------------


def test_ruff_key_uses_line_content_hash_not_line_number(fx: Path):
    spec = parsers.PARSERS["ruff"]
    keyed = ctb.key_findings(parsers.parse_ruff(RUFF_OUT), spec, fx)
    keys = [f.key() for f in keyed]
    assert keys[1] == f"pkg/mod.py::{parsers.line_hash('import os')}::F401"
    assert keys[2] == f"pkg/mod.py::{parsers.line_hash('import sys')}::F401"
    assert not any(":1:" in k or "::1::" in k for k in keys)
    # Shifting the file down by five lines and re-reporting at the new line
    # numbers yields the same keys.
    (fx / "pkg/mod.py").write_text("\n" * 5 + _SOURCE_FILES["pkg/mod.py"], encoding="utf-8")
    shifted = RUFF_OUT.replace("pkg/mod.py:1:8", "pkg/mod.py:6:8").replace(
        "pkg/mod.py:2:8", "pkg/mod.py:7:8"
    )
    assert [f.key() for f in ctb.key_findings(parsers.parse_ruff(shifted), spec, fx)] == keys


def test_keys_are_relative_posix_paths_even_when_tool_prints_absolute(fx: Path):
    spec = parsers.PARSERS["ruff"]
    absolute = RUFF_OUT.replace("pkg/mod.py:", f"{fx / 'pkg' / 'mod.py'}:")
    keyed = ctb.key_findings(parsers.parse_ruff(absolute), spec, fx)
    assert {f.path for f in keyed} == {"other.py", "pkg/mod.py"}
    dotted = ctb.key_findings(parsers.parse_todo(TODO_OUT), parsers.PARSERS["todo"], fx)
    assert {f.path for f in dotted} == {"other.py", "pkg/mod.py"}


# --- exit codes: 0 / 1 / 3 ----------------------------------------------------


def test_exit_code_0_when_baseline_equal_and_summary_names_counts(fx: Path, capsys):
    baseline = fx / "b.json"
    tool = _fake_tool(fx, RUFF_OUT, rc=1)
    assert _run("ruff", baseline, fx, tool, "--update") == 0
    assert _run("ruff", baseline, fx, tool) == 0
    out = capsys.readouterr().out
    assert "ruff: 0 new findings (3 baselined, 0 resolved)" in out
    assert "current 3 key(s) / 3 occurrence(s)" in out


def test_exit_code_1_on_new_finding_prints_key_baseline_path_and_remedies(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    before = baseline.read_bytes()
    (fx / "b.py").write_text("import json\n", encoding="utf-8")
    grown = RUFF_OUT + "b.py:1:8: F401 [*] `json` imported but unused\n"
    assert _run("ruff", baseline, fx, _fake_tool(fx, grown, rc=1)) == 1
    out = capsys.readouterr().out
    expected_key = f"b.py::{parsers.line_hash('import json')}::F401"
    assert f"NEW {expected_key}" in out
    assert str(baseline) in out
    assert "--update" in out and "--allow-grow" in out and "--reason" in out
    assert baseline.read_bytes() == before  # a failing check never rewrites


def test_exit_code_1_when_count_of_existing_key_grows(fx: Path):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    duplicated = RUFF_OUT + "pkg/mod.py:1:8: F401 [*] `os` imported but unused\n"
    assert _run("ruff", baseline, fx, _fake_tool(fx, duplicated, rc=1)) == 1


def test_exit_code_0_when_findings_resolved_and_resolved_count_printed(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    fewer = "\n".join(RUFF_OUT.splitlines()[1:]) + "\n"  # drop other.py F841
    assert _run("ruff", baseline, fx, _fake_tool(fx, fewer, rc=1)) == 0
    assert "ruff: 0 new findings (3 baselined, 1 resolved)" in capsys.readouterr().out


def test_exit_code_3_when_tool_not_found_and_baseline_untouched(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    before = baseline.read_bytes()
    assert _run("ruff", baseline, fx, ["/nonexistent-tool-xyz", "check", "."]) == 3
    err = capsys.readouterr().err
    assert "tool failed to run" in err
    assert baseline.read_bytes() == before


def test_exit_code_3_on_tool_crash_never_rewrites_even_with_update(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    before = baseline.read_bytes()
    crash = _fake_tool(fx, "", rc=2, stderr="error: unexpected argument '--nope'\n")
    assert _run("ruff", baseline, fx, crash) == 3
    assert _run("ruff", baseline, fx, crash, "--update") == 3
    assert _run("ruff", baseline, fx, crash, "--update", "--allow-grow", "--reason", "x") == 3
    err = capsys.readouterr().err
    assert "no parseable findings" in err and "left untouched" in err
    assert baseline.read_bytes() == before


def test_exit_code_3_only_when_no_findings_parsed(fx: Path):
    # A non-zero exit WITH parseable findings is the normal "tool found things" case.
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1)) == 0
    # A clean exit with empty stdout is zero findings, not a crash.
    assert _run("ruff", baseline, fx, _fake_tool(fx, "", rc=0)) == 0


# --- todo: grep semantics ----------------------------------------------------


def test_todo_grep_no_matches_exit_1_is_zero_findings_exit_0(fx: Path, capsys):
    baseline = fx / "todo.json"
    assert _run("todo", baseline, fx, _fake_tool(fx, "", rc=1), "--update") == 0
    assert json.loads(baseline.read_text())["findings"] == {}
    assert _run("todo", baseline, fx, _fake_tool(fx, "", rc=1)) == 0
    assert "todo: 0 new findings (0 baselined, 0 resolved)" in capsys.readouterr().out
    # grep exit 2 (usage / unreadable) with no output is still a crash.
    assert _run("todo", baseline, fx, _fake_tool(fx, "", rc=2)) == 3


def test_todo_new_fixme_in_new_file_exits_1_naming_it(fx: Path, capsys):
    baseline = fx / "todo.json"
    assert _run("todo", baseline, fx, _fake_tool(fx, TODO_OUT, rc=0), "--update") == 0
    grown = TODO_OUT + "./newfile.py:3:# FIXME: broken\n"
    assert _run("todo", baseline, fx, _fake_tool(fx, grown, rc=0)) == 1
    out = capsys.readouterr().out
    assert "newfile.py" in out and "::FIXME" in out


# --- mypy end to end ----------------------------------------------------------


def test_mypy_second_error_is_new_finding(fx: Path, capsys):
    baseline = fx / "mypy.json"
    assert _run("mypy", baseline, fx, _fake_tool(fx, MYPY_OUT, rc=1), "--update") == 0
    assert _run("mypy", baseline, fx, _fake_tool(fx, MYPY_OUT, rc=1)) == 0
    grown = MYPY_OUT + 'other.py:5: error: Name "y" is not defined  [name-defined]\n'
    assert _run("mypy", baseline, fx, _fake_tool(fx, grown, rc=1)) == 1
    assert "other.py::" in capsys.readouterr().out
    keys = json.loads(baseline.read_text())["findings"]
    assert all(k.startswith("pkg/mod.py::") for k in keys)
    assert {k.rsplit("::", 1)[1] for k in keys} == {"return-value", "arg-type", "assignment"}


# --- --update: shrink-only / subset rule ------------------------------------


def test_update_is_shrink_only_and_refuses_growth_without_allow_grow(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    before = baseline.read_bytes()
    grown = RUFF_OUT + "b.py:1:8: F401 [*] `json` imported but unused\n"
    assert _run("ruff", baseline, fx, _fake_tool(fx, grown, rc=1), "--update") == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err and "shrink-only" in captured.err
    assert "--allow-grow" in captured.err
    assert baseline.read_bytes() == before


def test_update_shrinks_to_strict_subset_of_previous_keys(fx: Path):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    previous = set(json.loads(baseline.read_text())["findings"])
    fewer = "\n".join(RUFF_OUT.splitlines()[1:]) + "\n"
    assert _run("ruff", baseline, fx, _fake_tool(fx, fewer, rc=1), "--update") == 0
    current = set(json.loads(baseline.read_text())["findings"])
    assert current < previous
    assert len(current) == 2


def test_allow_grow_requires_reason_and_records_it(fx: Path):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    grown = _fake_tool(fx, RUFF_OUT + "b.py:1:8: F401 [*] `json` imported but unused\n", rc=1)
    with pytest.raises(SystemExit) as exc:
        _run("ruff", baseline, fx, grown, "--update", "--allow-grow")
    assert exc.value.code == 2  # argparse usage error
    assert (
        _run(
            "ruff",
            baseline,
            fx,
            grown,
            "--update",
            "--allow-grow",
            "--reason",
            "mission fixture growth",
        )
        == 0
    )
    data = json.loads(baseline.read_text())
    assert len(data["findings"]) == 4
    assert data["growth_log"][-1]["reason"] == "mission fixture growth"
    assert data["growth_log"][-1]["added"] == 1
    assert "mission fixture growth" in baseline.read_text()
    # After growth the check is green again.
    assert _run("ruff", baseline, fx, grown) == 0


def test_update_is_idempotent_byte_identical_including_generated_at(fx: Path, monkeypatch):
    baseline = fx / "b.json"
    tool = _fake_tool(fx, RUFF_OUT, rc=1)
    assert _run("ruff", baseline, fx, tool, "--update") == 0
    first = baseline.read_bytes()
    monkeypatch.setattr(ctb, "_utc_now", lambda: "2099-01-01T00:00:00Z")
    assert _run("ruff", baseline, fx, tool, "--update") == 0
    assert baseline.read_bytes() == first


def test_update_creates_missing_baseline_and_check_requires_one(fx: Path, capsys):
    baseline = fx / "new" / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1)) == 2
    assert "baseline not found" in capsys.readouterr().err
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    assert baseline.exists()


# --- baseline shape and errors (exit 2) ---------------------------------------


def test_baseline_json_shape_sorted_keys_relative_paths(fx: Path):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    data = json.loads(baseline.read_text())
    assert data["tool"] == "ruff"
    assert data["version"] == 1
    assert isinstance(data["generated_at"], str) and data["generated_at"].endswith("Z")
    assert isinstance(data["findings"], dict)
    keys = list(data["findings"])
    assert keys == sorted(keys)
    assert all(not k.startswith(("/", "./")) for k in keys)
    assert all(isinstance(v, int) for v in data["findings"].values())
    assert list(data) == ["tool", "version", "generated_at", "findings"]
    assert baseline.read_text().endswith("}\n")


def test_baseline_tool_mismatch_exits_2_naming_it(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 0
    assert _run("mypy", baseline, fx, _fake_tool(fx, MYPY_OUT, rc=1)) == 2
    err = capsys.readouterr().err
    assert "tool mismatch" in err and "baseline=ruff" in err and "requested=mypy" in err
    # A tool with no parser yet (e.g. vulture before m1-ratchet-parsers) is
    # still reported as a mismatch against a ruff baseline, never 0 or 1.
    assert _run("vulture", baseline, fx, _fake_tool(fx, "", rc=0)) == 2
    assert "tool mismatch" in capsys.readouterr().err


def test_unregistered_tool_exits_2_naming_supported_parsers(fx: Path, capsys):
    baseline = fx / "b.json"
    assert _run("nosuchtool", baseline, fx, _fake_tool(fx, "", rc=0), "--update") == 2
    err = capsys.readouterr().err
    assert "no parser registered" in err and "ruff" in err and "todo" in err
    assert not baseline.exists()


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        ("{not json", "invalid JSON"),
        ('{"tool": "ruff", "version": 1, "generated_at": ""}', "findings"),
        ('{"tool": "ruff", "version": 1, "findings": ["a"]}', "findings"),
        ('{"tool": "ruff", "version": 1, "findings": {"k": "1"}}', "findings"),
        ('{"version": 1, "findings": {}}', "tool"),
        ('{"tool": "ruff", "version": 99, "findings": {}}', "version"),
        ("[]", "not an object"),
    ],
)
def test_corrupt_baseline_exits_2_with_diagnostic(fx: Path, capsys, content: str, needle: str):
    baseline = fx / "b.json"
    baseline.write_text(content, encoding="utf-8")
    before = baseline.read_bytes()
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1)) == 2
    assert needle in capsys.readouterr().err
    # Even --update must not "repair" a corrupt file silently.
    assert _run("ruff", baseline, fx, _fake_tool(fx, RUFF_OUT, rc=1), "--update") == 2
    assert baseline.read_bytes() == before


# --- --report-json ------------------------------------------------------------


def test_report_json_lists_tool_and_new_keys(fx: Path):
    baseline = fx / "b.json"
    report = fx / "report.json"
    tool = _fake_tool(fx, RUFF_OUT, rc=1)
    assert _run("ruff", baseline, fx, tool, "--update") == 0
    assert _run("ruff", baseline, fx, tool, "--report-json", str(report)) == 0
    clean = json.loads(report.read_text())
    assert clean["tool"] == "ruff" and clean["new_findings"] == [] and clean["exit_code"] == 0
    assert clean["baselined_count"] == 3 and clean["current_keys"] == 3
    (fx / "b.py").write_text("import json\n", encoding="utf-8")
    grown = RUFF_OUT + "b.py:1:8: F401 [*] `json` imported but unused\n"
    assert (
        _run("ruff", baseline, fx, _fake_tool(fx, grown, rc=1), "--report-json", str(report)) == 1
    )
    failed = json.loads(report.read_text())
    assert failed["new_findings"] == [f"b.py::{parsers.line_hash('import json')}::F401"]
    assert failed["new_count"] == 1 and failed["exit_code"] == 1
    assert failed["baseline"] == str(baseline)


# --- CLI surface --------------------------------------------------------------


def test_help_documents_every_flag_and_exit_codes(capsys):
    with pytest.raises(SystemExit) as exc:
        ctb.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--tool",
        "--baseline",
        "--update",
        "--allow-grow",
        "--reason",
        "--report-json",
        "--cwd",
        "-- <command",
    ):
        assert flag in out, flag
    for name in ("ruff", "mypy", "todo"):
        assert name in out, name
    assert "exit codes" in out and "3 tool failed" in out


def test_missing_command_or_bad_flag_combo_is_usage_error(fx: Path):
    baseline = fx / "b.json"
    with pytest.raises(SystemExit) as exc:
        ctb.main(["--tool", "ruff", "--baseline", str(baseline)])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        ctb.main(
            [
                "--tool",
                "ruff",
                "--baseline",
                str(baseline),
                "--allow-grow",
                "--reason",
                "r",
                "--",
                "x",
            ]
        )
    assert exc.value.code == 2  # --allow-grow without --update
    with pytest.raises(SystemExit) as exc:
        ctb.main(["--tool", "ruff", "--baseline", str(baseline), "--reason", "r", "--", "x"])
    assert exc.value.code == 2


def test_runner_strips_ansi_colour_from_tool_output(fx: Path):
    baseline = fx / "b.json"
    coloured = RUFF_OUT.replace("pkg/mod.py", "\x1b[1mpkg/mod.py\x1b[0m").replace(
        "F401", "\x1b[1m\x1b[31mF401\x1b[0m"
    )
    assert _run("ruff", baseline, fx, _fake_tool(fx, coloured, rc=1), "--update") == 0
    keys = json.loads(baseline.read_text())["findings"]
    assert len(keys) == 3 and all("\x1b" not in k for k in keys)
