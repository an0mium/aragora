"""Executable contract traces for the agent operating loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "orientation.v1.json"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = {
    "fresh_orientation.json",
    "interrupted_resumption.json",
    "quiet_no_change.json",
    "uncertain_high_risk.json",
}
DERIVED_COLLECTIONS = (
    "work_recommendations",
    "beliefs",
    "questions",
    "affordances",
    "obligations",
)
DERIVED_METADATA = {
    "basis_fingerprint",
    "evidence_refs",
    "authority",
    "freshness",
    "invalidators",
    "bounded_cost",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator() -> Any:
    schema = _load(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


@pytest.mark.parametrize("fixture_name", sorted(FIXTURE_NAMES))
def test_trace_conforms_to_orientation_v1(validator: Any, fixture_name: str) -> None:
    validator.validate(_load(FIXTURE_DIR / fixture_name))


def test_trace_set_is_exact() -> None:
    assert {path.name for path in FIXTURE_DIR.glob("*.json")} == FIXTURE_NAMES


@pytest.mark.parametrize("fixture_name", sorted(FIXTURE_NAMES - {"quiet_no_change.json"}))
def test_derived_records_preserve_lower_layer_basis(fixture_name: str) -> None:
    document = _load(FIXTURE_DIR / fixture_name)
    for collection in DERIVED_COLLECTIONS:
        for record in document[collection]:
            assert DERIVED_METADATA <= record.keys()
            assert "reasoning_fingerprint" not in record


def test_live_blocker_overrides_ready_recommendation() -> None:
    document = _load(FIXTURE_DIR / "interrupted_resumption.json")
    assert document["work_recommendations"][0]["classification"] == "ready"
    affordance = document["affordances"][0]
    assert affordance["authority"] == "live_authority"
    assert affordance["disposition"] == "blocked"
    assert affordance["blocked_by"] == ["settlement:BLOCKED"]


def test_high_risk_trace_requests_authorization_without_effect() -> None:
    document = _load(FIXTURE_DIR / "uncertain_high_risk.json")
    affordance = document["affordances"][0]
    assert affordance["risk_tier"] == 4
    assert affordance["disposition"] == "requires_authorization"
    assert document["mutations"] == []


def test_quiet_no_change_trace_fits_wire_budget() -> None:
    document = _load(FIXTURE_DIR / "quiet_no_change.json")
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= 800
    assert document["orientation_fingerprint"] == document["since_fingerprint"]
    assert document["mutations"] == []


@pytest.mark.parametrize(
    "doc_path",
    [
        "docs/THESIS.md",
        "docs/plans/ARAGORA_EVOLUTION_ROADMAP.md",
        "docs/AGENT_FLYWHEEL_ARAGORA_NATIVE.md",
        "docs/plans/2026-06-25-native-mission-orchestrator-spec.md",
        "docs/architecture/nomic-context-builder-plan.md",
    ],
)
def test_canonical_sources_link_to_operating_loop(doc_path: str) -> None:
    assert "agent-operating-loop.md" in (ROOT / doc_path).read_text(encoding="utf-8")
