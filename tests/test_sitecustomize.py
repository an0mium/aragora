from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib


def _load_repo_sitecustomize():
    module_path = Path(__file__).resolve().parents[1] / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("repo_sitecustomize", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detects_pytest_module_invocation() -> None:
    sitecustomize = _load_repo_sitecustomize()
    assert sitecustomize._is_pytest_invocation(["python", "-m", "pytest", "tests/"])


def test_detects_pytest_entrypoint_invocation() -> None:
    sitecustomize = _load_repo_sitecustomize()
    assert sitecustomize._is_pytest_invocation(["/usr/local/bin/pytest", "tests/"])


def test_configure_pytest_startup_removes_all_user_site_entries() -> None:
    sitecustomize = _load_repo_sitecustomize()
    path = ["/repo", "/tmp/user-site", "/tmp/user-site", "/stdlib"]

    sitecustomize._configure_pytest_startup(
        argv=["pytest", "tests/server/test_debate_origin.py"],
        path=path,
        user_site="/tmp/user-site",
    )

    assert "/tmp/user-site" not in path


def test_pyproject_disables_rerunfailures_by_default() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert "-p no:rerunfailures" in addopts
