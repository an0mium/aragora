"""Focused tests for the Boundary 2 fail-on-new checker."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "ci" / "check_boundary_edges.py"

_spec = importlib.util.spec_from_file_location("check_boundary_edges", CHECKER_PATH)
assert _spec and _spec.loader
cbe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cbe
_spec.loader.exec_module(cbe)


def _write_file(root: Path, relative: str, content: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_map(root: Path) -> Path:
    paths = [
        "aragora/gauntlet/receipt.py",
        "aragora/gauntlet/odr_export.py",
        "aragora/receipts",
        "aragora/gauntlet/odr_schema.json",
        "aragora-verify",
        "docs/specs/OPEN_DECISION_RECEIPT.md",
    ]
    for relative in paths:
        path = root / relative
        if Path(relative).suffix:
            _write_file(root, relative, "{}\n" if relative.endswith(".json") else "")
        else:
            path.mkdir(parents=True, exist_ok=True)
    _write_file(root, "aragora/receipts/__init__.py")
    _write_file(root, "aragora/core/__init__.py")
    _write_file(root, "aragora/server/__init__.py")
    _write_file(root, "aragora/server/handlers.py")
    _write_file(root, "aragora-verify/src/aragora_verify/__init__.py")
    _write_file(root, "aragora-verify/src/aragora_verify/odr_schema.json", "{}\n")
    _write_file(
        root,
        "aragora-verify/pyproject.toml",
        '[project]\nname = "aragora-verify"\ndependencies = ["cryptography>=48.0.1"]\n',
    )

    data = {
        "schema_version": 1,
        "boundary": {
            "id": 2,
            "name": "receipts+verifier",
            "provenance": "docs/architecture/MODULE_QUARANTINE_PROPOSAL.md#boundary-2",
        },
        "members": paths,
        "python_import_policy": {
            "sources": [
                "aragora/gauntlet/receipt.py",
                "aragora/gauntlet/odr_export.py",
                "aragora/receipts",
                "aragora-verify/src/aragora_verify",
            ],
            "allowed_internal_prefixes": [
                "aragora.core",
                "aragora.gauntlet.receipt",
                "aragora.receipts",
            ],
            "standalone": {
                "source": "aragora-verify/src/aragora_verify",
                "project_file": "aragora-verify/pyproject.toml",
                "package_root": "aragora_verify",
                "allowed_external_roots": ["cryptography"],
            },
        },
        "mirror_pairs": [
            {
                "left": "aragora/gauntlet/odr_schema.json",
                "right": "aragora-verify/src/aragora_verify/odr_schema.json",
            }
        ],
    }
    path = _write_file(root, "boundary.json", json.dumps(data))
    return path


def _write_baseline(
    root: Path,
    map_path: Path,
    violations: set[str],
    *,
    frozen_from_ref: str | None = None,
) -> Path:
    path = root / "baseline.json"
    data = {
        "schema_version": 1,
        "boundary": {"id": 2, "name": "receipts+verifier"},
        "map": str(map_path.relative_to(root)),
        "violations": sorted(violations),
    }
    if frozen_from_ref is not None:
        data["frozen_from_ref"] = frozen_from_ref
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )
    return path


def _args(root: Path, map_path: Path, baseline: Path) -> list[str]:
    return [
        "--repo-root",
        str(root),
        "--map",
        str(map_path),
        "--baseline",
        str(baseline),
    ]


def test_new_forbidden_edge_exits_one_and_names_offender(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora/gauntlet/receipt.py",
        "from aragora.server import handlers\n",
    )

    rc = cbe.main(_args(tmp_path, map_path, baseline))

    assert rc == 1
    assert "import aragora.gauntlet.receipt -> aragora.server.handlers" in capsys.readouterr().out


def test_allowed_package_importfrom_resolves_submodule(tmp_path):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora/gauntlet/receipt.py",
        "from aragora import core\n",
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 0


def test_forbidden_package_importfrom_names_submodule(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora/gauntlet/receipt.py",
        "from aragora import server\n",
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 1
    assert "import aragora.gauntlet.receipt -> aragora.server" in capsys.readouterr().out


def test_dotted_importfrom_is_not_absorbed_by_parent_baseline(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(
        tmp_path,
        map_path,
        {"import aragora.gauntlet.odr_export -> aragora.gauntlet"},
    )
    _write_file(tmp_path, "aragora/gauntlet/runner.py")
    _write_file(
        tmp_path,
        "aragora/gauntlet/odr_export.py",
        "from aragora.gauntlet import runner\n",
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 1
    output = capsys.readouterr().out
    assert "import aragora.gauntlet.odr_export -> aragora.gauntlet.runner" in output


def test_relative_importfrom_uses_known_submodule(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(tmp_path, "aragora/gauntlet/runner.py")
    _write_file(
        tmp_path,
        "aragora/gauntlet/receipt.py",
        "from . import runner\n",
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 1
    assert "import aragora.gauntlet.receipt -> aragora.gauntlet.runner" in capsys.readouterr().out


def test_baselined_edge_and_resolved_subset_exit_zero(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(
        tmp_path,
        map_path,
        {
            "import aragora.gauntlet.receipt -> aragora.server.handlers",
            "offline aragora_verify -> jsonschema",
        },
    )
    _write_file(
        tmp_path,
        "aragora/gauntlet/receipt.py",
        "from aragora.server import handlers\n",
    )

    rc = cbe.main(_args(tmp_path, map_path, baseline))

    assert rc == 0
    output = capsys.readouterr().out
    assert "no new" in output.lower()
    assert "1 baselined violation(s) are resolved" in output


def test_standalone_dependency_is_offline_violation(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora-verify/src/aragora_verify/schema.py",
        "import jsonschema\n",
    )

    rc = cbe.main(_args(tmp_path, map_path, baseline))

    assert rc == 1
    assert "offline aragora_verify.schema -> jsonschema" in capsys.readouterr().out


def test_declared_standalone_dependency_is_offline_violation(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora-verify/pyproject.toml",
        '[project]\nname = "aragora-verify"\ndependencies = ["httpx[socks]>=0.28"]\n',
    )

    rc = cbe.main(_args(tmp_path, map_path, baseline))

    assert rc == 1
    assert "offline dependency aragora-verify -> httpx" in capsys.readouterr().out


def test_optional_standalone_dependency_is_offline_violation(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora-verify/pyproject.toml",
        (
            '[project]\nname = "aragora-verify"\ndependencies = []\n'
            '[project.optional-dependencies]\nschema = ["httpx[socks]>=0.28"]\n'
        ),
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 1
    assert "offline dependency aragora-verify -> httpx" in capsys.readouterr().out


def test_unreachable_frozen_ref_is_informational_only(tmp_path):
    map_path = _write_map(tmp_path)
    _write_file(
        tmp_path,
        "aragora-verify/pyproject.toml",
        (
            '[project]\nname = "aragora-verify"\ndependencies = []\n'
            '[project.optional-dependencies]\nschema = ["jsonschema>=4.0"]\n'
        ),
    )
    baseline = _write_baseline(
        tmp_path,
        map_path,
        {"offline dependency aragora-verify -> jsonschema"},
        frozen_from_ref="0000000000000000000000000000000000000000",
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 0


@pytest.mark.parametrize("dependency", ["jsonschema", "pytest"])
def test_removing_dependency_from_baseline_makes_it_new(tmp_path, capsys, dependency):
    map_path = _write_map(tmp_path)
    _write_file(
        tmp_path,
        "aragora-verify/pyproject.toml",
        (
            '[project]\nname = "aragora-verify"\ndependencies = []\n'
            "[project.optional-dependencies]\n"
            'schema = ["jsonschema>=4.0"]\n'
            'dev = ["pytest>=8.0"]\n'
        ),
    )
    all_dependency_edges = {
        "offline dependency aragora-verify -> jsonschema",
        "offline dependency aragora-verify -> pytest",
    }
    missing_edge = f"offline dependency aragora-verify -> {dependency}"
    baseline = _write_baseline(
        tmp_path,
        map_path,
        all_dependency_edges - {missing_edge},
    )

    assert cbe.main(_args(tmp_path, map_path, baseline)) == 1
    assert missing_edge in capsys.readouterr().out


def test_boundary_hook_covers_default_install_and_nested_sources():
    precommit = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hook = next(
        hook
        for repository in precommit["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "boundary2-edges"
    )
    boundary = cbe.load_boundary_map(REPO_ROOT, cbe.DEFAULT_MAP)
    pattern = re.compile(hook["files"])

    assert set(hook["stages"]) == {"pre-commit", "pre-push"}
    assert hook["files"] == cbe.expected_hook_files_pattern(REPO_ROOT, boundary)
    assert pattern.search("aragora/receipts/__init__.py")
    assert pattern.search("aragora-verify/src/aragora_verify/verifier.py")
    assert pattern.search("aragora-verify/pyproject.toml")
    assert pattern.search("aragora-verify/src/aragora_verify/odr_schema.json")
    assert pattern.search("docs/specs/OPEN_DECISION_RECEIPT.md")
    assert pattern.search("aragora/new_module.py")
    assert not pattern.search("frontend/src/unrelated.ts")


def test_schema_mirror_drift_exits_one(tmp_path, capsys):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    _write_file(
        tmp_path,
        "aragora-verify/src/aragora_verify/odr_schema.json",
        '{"changed": true}\n',
    )

    rc = cbe.main(_args(tmp_path, map_path, baseline))

    assert rc == 1
    assert (
        "mirror aragora/gauntlet/odr_schema.json != "
        "aragora-verify/src/aragora_verify/odr_schema.json"
    ) in capsys.readouterr().out


def test_freeze_refuses_growth_and_preserves_baseline(tmp_path):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(tmp_path, map_path, set())
    original = baseline.read_text(encoding="utf-8")
    _write_file(
        tmp_path,
        "aragora/gauntlet/receipt.py",
        "from aragora.server import handlers\n",
    )

    rc = cbe.main([*_args(tmp_path, map_path, baseline), "--freeze"])

    assert rc == 2
    assert baseline.read_text(encoding="utf-8") == original


def test_freeze_shrinks_without_adopt(tmp_path):
    map_path = _write_map(tmp_path)
    baseline = _write_baseline(
        tmp_path,
        map_path,
        {"import aragora.gauntlet.receipt -> aragora.server"},
    )

    rc = cbe.main([*_args(tmp_path, map_path, baseline), "--freeze"])

    assert rc == 0
    config = cbe.load_boundary_map(tmp_path, map_path)
    assert cbe.load_baseline(baseline, config, tmp_path) == set()


def test_initial_freeze_requires_explicit_adopt(tmp_path):
    map_path = _write_map(tmp_path)
    baseline = tmp_path / "missing.json"

    assert cbe.main([*_args(tmp_path, map_path, baseline), "--freeze"]) == 2
    assert not baseline.exists()
    assert cbe.main([*_args(tmp_path, map_path, baseline), "--freeze", "--adopt"]) == 0
    assert baseline.exists()
