import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "check_capability_matrix_sync.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_capability_matrix_sync", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_paths(module, tmp_path: Path) -> Path:
    target = tmp_path / "docs" / "CAPABILITY_MATRIX.md"
    target.parent.mkdir(parents=True)
    target.write_text("current\n", encoding="utf-8")
    module.REPO_ROOT = tmp_path
    module.GEN_SCRIPT = tmp_path / "scripts" / "generate_capability_matrix.py"
    module.TARGETS = [target]
    return target


def test_main_reports_generator_failure_without_traceback(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_module()
    _install_paths(module, tmp_path)

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["generate_capability_matrix.py"],
            returncode=2,
            stdout="generator stdout",
            stderr="generator stderr",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() == 1

    captured = capsys.readouterr()
    assert (
        "Could not regenerate capability matrix target: docs/CAPABILITY_MATRIX.md" in captured.err
    )
    assert "generator stderr" in captured.err
    assert "generator stdout" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_up_to_date_when_generated_target_matches(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_module()
    target = _install_paths(module, tmp_path)

    def fake_run(cmd, *_args, **_kwargs):
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Capability matrix files are up to date." in captured.out
