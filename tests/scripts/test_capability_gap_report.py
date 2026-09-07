from __future__ import annotations

from pathlib import Path

import pytest

import scripts.capability_gap_report as gap_report


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _write_minimal_sources(repo_root: Path) -> None:
    _write(
        repo_root / "aragora" / "capabilities.yaml",
        """
capabilities:
  inbox_triage:
    name: Inbox triage
    category: workflow
    status: active
""",
    )
    _write(
        repo_root / "aragora" / "capability_surfaces.yaml",
        """
capabilities:
  inbox_triage:
    cli:
      - aragora inbox triage
    sdk:
      python:
        - aragora_sdk.inbox
""",
    )


def test_build_report_counts_valid_capability_sources(tmp_path: Path) -> None:
    _write_minimal_sources(tmp_path)

    report = gap_report.build_report(tmp_path)

    assert report["total_capabilities"] == 1
    assert report["mapped_capabilities"] == 1
    assert report["items"]["inbox_triage"]["name"] == "Inbox triage"
    assert report["gaps"]["api"] == ["inbox_triage"]
    assert report["gaps"]["sdk"] == []


def test_load_yaml_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Capability report YAML source missing"):
        gap_report._load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_rejects_empty_source(tmp_path: Path) -> None:
    source = _write(tmp_path / "empty.yaml", "")

    with pytest.raises(ValueError, match="is empty"):
        gap_report._load_yaml(source)


def test_load_yaml_rejects_malformed_source(tmp_path: Path) -> None:
    source = _write(tmp_path / "bad.yaml", "capabilities:\n  - [")

    with pytest.raises(ValueError, match="Malformed capability report YAML source"):
        gap_report._load_yaml(source)


def test_load_yaml_rejects_non_mapping_source(tmp_path: Path) -> None:
    source = _write(tmp_path / "list.yaml", "- inbox_triage\n")

    with pytest.raises(ValueError, match="must be a mapping"):
        gap_report._load_yaml(source)


def test_build_report_rejects_missing_catalog_before_empty_report(tmp_path: Path) -> None:
    _write(
        tmp_path / "aragora" / "capability_surfaces.yaml",
        """
capabilities:
  inbox_triage:
    cli:
      - aragora inbox triage
""",
    )

    with pytest.raises(FileNotFoundError, match="capabilities.yaml"):
        gap_report.build_report(tmp_path)
