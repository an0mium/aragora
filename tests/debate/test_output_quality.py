"""Tests for deterministic post-consensus output quality validation."""

from __future__ import annotations

import json

from aragora.debate.output_quality import (
    OutputContract,
    apply_deterministic_quality_repairs,
    build_concretization_prompt,
    build_upgrade_prompt,
    derive_output_contract_from_task,
    finalize_json_payload,
    load_output_contract_from_file,
    output_contract_from_dict,
    validate_output_against_contract,
)


def test_derive_output_contract_from_task_sections():
    task = (
        "Smoke test: output sections Ranked High-Level Tasks, Suggested Subtasks, "
        "Owner module / file paths, Test Plan, Rollback Plan, Gate Criteria, JSON Payload"
    )
    contract = derive_output_contract_from_task(task)
    assert contract is not None
    assert contract.required_sections == [
        "Ranked High-Level Tasks",
        "Suggested Subtasks",
        "Owner module / file paths",
        "Test Plan",
        "Rollback Plan",
        "Gate Criteria",
        "JSON Payload",
    ]
    assert contract.require_gate_thresholds is True
    assert contract.require_rollback_triggers is True
    assert contract.require_owner_paths is True
    assert contract.require_json_payload is True


def test_derive_output_contract_from_task_markdown_headers_phrase():
    task = (
        "Output MUST include exactly these sections as markdown headers: "
        "Ranked High-Level Tasks, Suggested Subtasks, Owner module / file paths, "
        "Test Plan, Rollback Plan, Gate Criteria, JSON Payload."
    )
    contract = derive_output_contract_from_task(task)
    assert contract is not None
    assert contract.required_sections == [
        "Ranked High-Level Tasks",
        "Suggested Subtasks",
        "Owner module / file paths",
        "Test Plan",
        "Rollback Plan",
        "Gate Criteria",
        "JSON Payload",
    ]


def test_derive_output_contract_from_task_fallback_known_headings():
    task = (
        "Return a plan with Ranked High-Level Tasks and Suggested Subtasks, then "
        "include Owner module / file paths, Test Plan, Rollback Plan, Gate Criteria, "
        "and JSON Payload."
    )
    contract = derive_output_contract_from_task(task)
    assert contract is not None
    assert contract.required_sections == [
        "Ranked High-Level Tasks",
        "Suggested Subtasks",
        "Owner module / file paths",
        "Test Plan",
        "Rollback Plan",
        "Gate Criteria",
        "JSON Payload",
    ]


def test_validate_output_against_contract_good_report():
    contract = OutputContract(
        required_sections=[
            "Ranked High-Level Tasks",
            "Suggested Subtasks",
            "Owner module / file paths",
            "Test Plan",
            "Rollback Plan",
            "Gate Criteria",
            "JSON Payload",
        ]
    )
    answer = """
## Ranked High-Level Tasks
- T1

## Suggested Subtasks
- S1

## Owner module / file paths
- aragora/cli/commands/debate.py
- tests/debate/test_output_quality.py

## Test Plan
- Unit tests

## Rollback Plan
If error_rate > 2%, rollback by disabling feature flag and redeploy previous image.

## Gate Criteria
- p95_latency <= 250ms for 10m
- error_rate < 1% over 15m

## JSON Payload
```json
{
  "ranked_high_level_tasks": ["T1"],
  "suggested_subtasks": ["S1"],
  "owner_module_file_paths": ["aragora/cli/commands/debate.py"],
  "test_plan": ["Unit tests"],
  "rollback_plan": {"trigger": "error_rate > 2%", "action": "disable flag"},
  "gate_criteria": [
    {"metric": "p95_latency", "op": "<=", "threshold": 250, "unit": "ms"},
    {"metric": "error_rate", "op": "<", "threshold": 1, "unit": "%"}
  ]
}
```
"""
    report = validate_output_against_contract(answer, contract)
    assert report.verdict == "good"
    assert report.has_gate_thresholds is True
    assert report.has_rollback_trigger is True
    assert report.has_paths is True
    assert report.has_valid_json_payload is True
    assert report.path_existence_rate >= 0.99
    assert report.practicality_score_10 >= 6.0
    assert report.defects == []


def test_validate_output_against_contract_detects_threshold_gap():
    contract = OutputContract(
        required_sections=["Gate Criteria", "Rollback Plan", "JSON Payload"],
        require_owner_paths=False,
    )
    answer = """
## Gate Criteria
- Keep it safe and reliable.

## Rollback Plan
If failure spikes then rollback immediately.

## JSON Payload
```json
{"ok": true}
```
"""
    report = validate_output_against_contract(answer, contract)
    assert report.verdict == "needs_work"
    assert report.has_gate_thresholds is False
    assert any("quantitative thresholds" in defect for defect in report.defects)


def test_validate_output_against_contract_detects_weak_repo_grounding():
    contract = OutputContract(
        required_sections=[
            "Owner module / file paths",
            "Rollback Plan",
            "Gate Criteria",
            "JSON Payload",
        ]
    )
    answer = """
## Owner module / file paths
- aragora/this/path/does/not/exist.py

## Rollback Plan
If error_rate > 2% for 10m, rollback by disabling the feature flag.

## Gate Criteria
- p95_latency <= 250ms for 15m
- error_rate < 1% over 15m

## JSON Payload
```json
{"ok": true}
```
"""
    report = validate_output_against_contract(answer, contract)
    assert report.verdict == "needs_work"
    assert report.path_existence_rate == 0.0
    assert any("weakly grounded" in defect for defect in report.defects)


def test_build_upgrade_prompt_contains_defects_and_contract():
    contract = OutputContract(required_sections=["A", "B"])
    prompt = build_upgrade_prompt(
        task="t",
        contract=contract,
        current_answer="old",
        defects=["Missing required section: B"],
    )
    assert "Missing required section: B" in prompt
    assert "1. A" in prompt
    assert "2. B" in prompt
    assert "Current answer" in prompt


def test_build_concretization_prompt_contains_practicality_target():
    contract = OutputContract(required_sections=["Ranked High-Level Tasks", "JSON Payload"])
    prompt = build_concretization_prompt(
        task="t",
        contract=contract,
        current_answer="old",
        practicality_score_10=4.2,
        target_practicality_10=7.0,
        defects=["Output practicality is too low for execution handoff."],
    )
    assert "Current practicality score (0-10): 4.2" in prompt
    assert "Target practicality score (0-10): 7.0" in prompt
    assert "Replace placeholders" in prompt


def test_apply_deterministic_quality_repairs_improves_report():
    contract = OutputContract(
        required_sections=[
            "Owner module / file paths",
            "Rollback Plan",
            "Gate Criteria",
            "JSON Payload",
        ]
    )
    weak = """
## Owner module / file paths
- TBD

## Rollback Plan
We will monitor.

## Gate Criteria
- Good quality

## JSON Payload
```json
{"bad": trailing,}
```
"""
    before = validate_output_against_contract(weak, contract)
    repaired = apply_deterministic_quality_repairs(weak, contract, before)
    after = validate_output_against_contract(repaired, contract)
    assert after.quality_score_10 >= before.quality_score_10
    assert after.has_valid_json_payload is True
    assert after.has_rollback_trigger is True


def test_output_contract_from_dict_parses_flags():
    contract = output_contract_from_dict(
        {
            "required_sections": ["A", "B"],
            "require_json_payload": "true",
            "require_gate_thresholds": "false",
            "require_rollback_triggers": True,
            "require_owner_paths": False,
            "require_repo_path_existence": False,
            "require_practicality_checks": False,
        }
    )
    assert contract.required_sections == ["A", "B"]
    assert contract.require_json_payload is True
    assert contract.require_gate_thresholds is False
    assert contract.require_rollback_triggers is True
    assert contract.require_owner_paths is False
    assert contract.require_repo_path_existence is False
    assert contract.require_practicality_checks is False


def test_load_output_contract_from_file(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "required_sections": [
                    "Ranked High-Level Tasks",
                    "JSON Payload",
                ],
                "require_json_payload": True,
                "require_gate_thresholds": False,
                "require_rollback_triggers": False,
                "require_owner_paths": False,
            }
        ),
        encoding="utf-8",
    )
    contract = load_output_contract_from_file(str(path))
    assert contract.required_sections == ["Ranked High-Level Tasks", "JSON Payload"]
    assert contract.require_json_payload is True
    assert contract.require_gate_thresholds is False


def test_finalize_json_payload_repairs_invalid_json_section():
    contract = OutputContract(
        required_sections=[
            "Ranked High-Level Tasks",
            "Gate Criteria",
            "JSON Payload",
        ],
        require_rollback_triggers=False,
        require_owner_paths=False,
    )
    answer = """
## Ranked High-Level Tasks
- Task A

## Gate Criteria
- error_rate < 1% over 15m
- p95_latency <= 250ms for 15m

## JSON Payload
```json
{"broken":
```
"""
    fixed = finalize_json_payload(answer, contract)
    report = validate_output_against_contract(fixed, contract)
    assert report.has_valid_json_payload is True
    assert report.verdict == "good"


def test_finalize_json_payload_includes_dissent_and_unresolved_risks():
    contract = OutputContract(
        required_sections=[
            "Ranked High-Level Tasks",
            "JSON Payload",
        ],
        require_gate_thresholds=False,
        require_rollback_triggers=False,
        require_owner_paths=False,
    )
    answer = """
## Ranked High-Level Tasks
- Task A

## Dissent
- Concern: model collapse

## Unresolved Risks
- Risk: missing external benchmark
"""
    fixed = finalize_json_payload(answer, contract)
    assert "## JSON Payload" in fixed
    report = validate_output_against_contract(fixed, contract)
    assert report.has_valid_json_payload is True
    assert report.verdict == "good"


def test_finalize_json_payload_replaces_heading_with_colon_and_dedupes():
    contract = OutputContract(
        required_sections=[
            "Ranked High-Level Tasks",
            "JSON Payload",
        ],
        require_gate_thresholds=False,
        require_rollback_triggers=False,
        require_owner_paths=False,
    )
    answer = """
## Ranked High-Level Tasks
- Task A

## JSON Payload:
```json
{"broken":
```

## JSON Payload
```json
{"also":"broken",}
```
"""
    fixed = finalize_json_payload(answer, contract)
    assert fixed.count("## JSON Payload") == 1
    report = validate_output_against_contract(fixed, contract)
    assert report.has_valid_json_payload is True
    assert report.verdict == "good"
