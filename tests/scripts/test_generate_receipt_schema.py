from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_receipt_schema.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_receipt_schema", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["generate_receipt_schema"] = module
    spec.loader.exec_module(module)
    return module


schema_mod = _load_module()


def test_python_type_to_json_maps_generic_containers() -> None:
    assert schema_mod._python_type_to_json("list[str]") == "array"
    assert schema_mod._python_type_to_json("list[ReceiptFinding]") == "array"
    assert schema_mod._python_type_to_json("dict[str, Any]") == "object"
    assert schema_mod._python_type_to_json("dict[str, Any] | None") == ["object", "null"]
    assert schema_mod._python_type_to_json("list[dict] | None") == ["array", "null"]


def test_export_decision_receipt_schema_preserves_container_types() -> None:
    schema = schema_mod.generate_decision_receipt_schema()
    properties = schema["properties"]

    assert properties["findings"]["type"] == "array"
    assert properties["mitigations"]["type"] == "array"
    assert properties["dissenting_views"]["type"] == "array"
    assert properties["verified_claims"]["type"] == "array"
    assert properties["cost_summary"]["type"] == ["object", "null"]


def test_gauntlet_receipt_schema_preserves_container_types() -> None:
    schema = schema_mod.generate_gauntlet_receipt_schema()
    properties = schema["properties"]

    assert properties["vulnerability_details"]["type"] == "array"
    assert properties["dissenting_views"]["type"] == "array"
    assert properties["provenance_chain"]["type"] == "array"
    assert properties["agent_responses"]["type"] == "array"
    assert properties["explainability"]["type"] == ["object", "null"]
    assert properties["settlement_metadata"]["type"] == ["object", "null"]
    assert properties["config_used"]["type"] == "object"
