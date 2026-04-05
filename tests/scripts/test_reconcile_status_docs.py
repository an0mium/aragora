"""Tests for scripts/reconcile_status_docs.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "reconcile_status_docs.py"
    spec = importlib.util.spec_from_file_location("reconcile_status_docs", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load reconcile_status_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_placeholder_connector(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '"""\nThis connector is a placeholder for licensed integrations.\n"""\n',
        encoding="utf-8",
    )


def _write_connector_status(path: Path, *, stub_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Connector Status Matrix",
                "",
                "## Summary",
                "",
                "- **Production**: 149 connectors",
                "- **Beta**: 0 connectors",
                f"- **Stub**: {stub_count} connectors",
            ]
        ),
        encoding="utf-8",
    )


def test_check_connector_status_skips_warning_when_stub_count_matches_placeholders(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    connector_root = tmp_path / "aragora" / "connectors"
    status_doc = tmp_path / "docs" / "connectors" / "STATUS.md"

    _write_placeholder_connector(connector_root / "legal" / "lexis.py")
    _write_placeholder_connector(connector_root / "legal" / "westlaw.py")
    _write_connector_status(status_doc, stub_count=2)

    with (
        patch.object(module, "CONNECTOR_ROOT", connector_root),
        patch.object(module, "CONNECTOR_STATUS", status_doc),
    ):
        findings = module._check_connector_status()

    assert not any(finding["severity"] == "warning" for finding in findings)
    assert any("documented placeholder connectors" in finding["message"] for finding in findings)


def test_check_connector_status_warns_when_doc_stub_count_exceeds_placeholders(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    connector_root = tmp_path / "aragora" / "connectors"
    status_doc = tmp_path / "docs" / "connectors" / "STATUS.md"

    _write_placeholder_connector(connector_root / "legal" / "lexis.py")
    _write_connector_status(status_doc, stub_count=2)

    with (
        patch.object(module, "CONNECTOR_ROOT", connector_root),
        patch.object(module, "CONNECTOR_STATUS", status_doc),
    ):
        findings = module._check_connector_status()

    assert any(
        finding["severity"] == "warning"
        and "only 1 explicit placeholder connectors exist in code" in finding["message"]
        for finding in findings
    )
