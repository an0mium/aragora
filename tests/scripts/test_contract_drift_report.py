from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "contract_drift_report.py"


def _load_module() -> Any:
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("contract_drift_report", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["contract_drift_report"] = module
    spec.loader.exec_module(module)
    return module


contract_drift_report = _load_module()


def test_load_json_returns_object_payload(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"missing_from_both_sdks": ["GET /v1/a"]}), encoding="utf-8")

    assert contract_drift_report._load_json(baseline) == {"missing_from_both_sdks": ["GET /v1/a"]}


def test_load_json_rejects_missing_baseline(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    try:
        contract_drift_report._load_json(missing)
    except contract_drift_report.BaselineJsonError as exc:
        assert "baseline missing" in str(exc)
        assert str(missing) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing baseline should fail closed")


def test_load_json_rejects_malformed_baseline(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")

    try:
        contract_drift_report._load_json(malformed)
    except contract_drift_report.BaselineJsonError as exc:
        assert "cannot load contract drift baseline" in str(exc)
        assert str(malformed) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("malformed baseline should fail closed")


def test_load_json_rejects_non_object_baseline(tmp_path: Path) -> None:
    non_object = tmp_path / "baseline.json"
    non_object.write_text(json.dumps(["GET /v1/a"]), encoding="utf-8")

    try:
        contract_drift_report._load_json(non_object)
    except contract_drift_report.BaselineJsonError as exc:
        assert "must be a JSON object" in str(exc)
        assert str(non_object) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-object baseline should fail closed")


def test_main_stops_before_writing_reports_when_baseline_is_untrusted(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    json_out = tmp_path / "contract-drift-summary.json"
    md_out = tmp_path / "contract-drift-summary.md"

    def fail_summary() -> dict[str, Any]:
        raise contract_drift_report.BaselineJsonError(
            "cannot load contract drift baseline scripts/baselines/check_sdk_parity.json"
        )

    monkeypatch.setattr(contract_drift_report, "build_summary", fail_summary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "contract_drift_report.py",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
    )

    assert contract_drift_report.main() == 2
    captured = capsys.readouterr()
    assert "cannot load contract drift baseline" in captured.err
    assert not json_out.exists()
    assert not md_out.exists()
