import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "capability_gap_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capability_gap_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_yaml(root: Path, *, capabilities: str, surfaces: str) -> None:
    aragora_dir = root / "aragora"
    aragora_dir.mkdir()
    (aragora_dir / "capabilities.yaml").write_text(capabilities, encoding="utf-8")
    (aragora_dir / "capability_surfaces.yaml").write_text(surfaces, encoding="utf-8")


def test_build_report_accepts_minimal_capability_maps(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="""
capabilities:
  example:
    name: Example Capability
    category: proof
    status: stable
""",
        surfaces="""
capabilities:
  example:
    cli:
      - aragora example
    api: []
    sdk:
      python: []
      typescript: []
    ui: []
    channels: []
""",
    )

    report = module.build_report(tmp_path)

    assert report["total_capabilities"] == 1
    assert report["mapped_capabilities"] == 1
    assert report["items"]["example"]["name"] == "Example Capability"


def test_build_report_rejects_non_mapping_yaml_root(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="- not-a-map\n",
        surfaces="capabilities: {}\n",
    )

    with pytest.raises(ValueError, match="capabilities.yaml must contain a YAML mapping"):
        module.build_report(tmp_path)


def test_build_report_rejects_non_mapping_capability_catalog(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="capabilities: []\n",
        surfaces="capabilities: {}\n",
    )

    with pytest.raises(ValueError, match="capabilities catalog must be a mapping"):
        module.build_report(tmp_path)


def test_build_report_rejects_non_mapping_capability_entry(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="""
capabilities:
  example: stable
""",
        surfaces="capabilities: {}\n",
    )

    with pytest.raises(ValueError, match="capability catalog entry example must be a mapping"):
        module.build_report(tmp_path)


def test_build_report_rejects_non_mapping_surface_map(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="""
capabilities:
  example:
    name: Example
""",
        surfaces="capabilities: []\n",
    )

    with pytest.raises(ValueError, match="capability surface map must be a mapping"):
        module.build_report(tmp_path)


def test_build_report_rejects_non_mapping_surface_entry(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="""
capabilities:
  example:
    name: Example
""",
        surfaces="""
capabilities:
  example: exposed
""",
    )

    with pytest.raises(ValueError, match="capability surface entry must be a mapping"):
        module.build_report(tmp_path)


def test_build_report_rejects_non_mapping_surface_sdk_entry(tmp_path: Path) -> None:
    module = _load_module()
    _write_yaml(
        tmp_path,
        capabilities="""
capabilities:
  example:
    name: Example
""",
        surfaces="""
capabilities:
  example:
    sdk: python-client
""",
    )

    with pytest.raises(ValueError, match="capability surface sdk entry must be a mapping"):
        module.build_report(tmp_path)
