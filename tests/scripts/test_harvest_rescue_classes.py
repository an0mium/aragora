from __future__ import annotations

import json
import sys
from pathlib import Path

from aragora.swarm.rescue_events import RescueEvent, RescueEventLedger

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import harvest_rescue_classes as mod  # noqa: E402


def _ledger_with_events(tmp_path: Path, events: list[RescueEvent]) -> RescueEventLedger:
    ledger = RescueEventLedger(path=tmp_path / "rescue_events.jsonl")
    for event in events:
        ledger.record(event)
    return ledger


def _write_productization_map(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_harvest_repeated_rescue_classes_includes_productization_fields(tmp_path: Path) -> None:
    ledger = _ledger_with_events(
        tmp_path,
        [
            RescueEvent(
                event_type="session_restart",
                reason="runner freshness gate blocked launch",
                issue_number=5514,
            ),
            RescueEvent(
                event_type="session_restart",
                reason="runner freshness gate blocked launch",
                issue_number=5514,
            ),
            RescueEvent(
                event_type="followup_prompt",
                reason="needs explicit next step from founder",
                issue_number=5512,
            ),
            RescueEvent(
                event_type="followup_prompt",
                reason="needs explicit next step from founder",
                issue_number=5515,
            ),
        ],
    )
    productization_map = mod.load_productization_map(
        _write_productization_map(
            tmp_path / "rescue_productization.json",
            [
                {
                    "class": "session_restart:runner freshness gate blocked launch",
                    "target_kind": "issue",
                    "target": "#5514",
                    "title": "use contract receipts for dispatch admission",
                }
            ],
        )
    )

    rows = mod.harvest_repeated_rescue_classes(
        ledger,
        threshold=2,
        recent_limit=20,
        example_limit=5,
        productization_map=productization_map,
    )

    by_class = {row["class"]: row for row in rows}
    assert (
        by_class["session_restart:runner freshness gate blocked launch"]["productization_status"]
        == "linked_issue"
    )
    assert (
        by_class["session_restart:runner freshness gate blocked launch"]["productization_target"]
        == "#5514"
    )
    assert (
        by_class["followup_prompt:needs explicit next step from founder"]["productization_status"]
        == "unlinked"
    )


def test_summarize_rescue_classes_separates_repeated_from_one_off_noise(tmp_path: Path) -> None:
    ledger = _ledger_with_events(
        tmp_path,
        [
            RescueEvent(event_type="manual_merge", reason="required review gate"),
            RescueEvent(event_type="manual_merge", reason="required review gate"),
            RescueEvent(event_type="issue_rewrite", reason="scope contradicted itself"),
            RescueEvent(event_type="permission_approval", reason="grant shell access"),
            RescueEvent(event_type="permission_approval", reason="grant shell access"),
            RescueEvent(event_type="permission_approval", reason="grant shell access"),
        ],
    )

    report = mod.summarize_rescue_classes(
        ledger,
        threshold=2,
        recent_limit=20,
        example_limit=5,
        one_off_limit=10,
        productization_map={},
    )

    assert report["summary"]["repeated_class_count"] == 2
    assert report["summary"]["one_off_class_count"] == 1
    assert report["summary"]["below_threshold_class_count"] == 0
    assert {row["class"] for row in report["repeated_classes"]} == {
        "permission_approval:grant shell access",
        "manual_merge:required review gate",
    }
    assert report["one_off_classes"] == [
        {
            "class": "issue_rewrite:scope contradicted itself",
            "count": 1,
            "event_type": "issue_rewrite",
            "productization_notes": "",
            "productization_status": "unlinked",
            "productization_target": "",
            "productization_target_kind": "",
            "productization_title": "",
            "reason_excerpt": "scope contradicted itself",
        }
    ]


def test_main_report_json_emits_summary_and_linkage(tmp_path: Path, capsys) -> None:
    ledger = _ledger_with_events(
        tmp_path,
        [
            RescueEvent(
                event_type="session_restart",
                reason="runner freshness gate blocked launch",
                issue_number=5514,
            ),
            RescueEvent(
                event_type="session_restart",
                reason="runner freshness gate blocked launch",
                issue_number=5514,
            ),
            RescueEvent(event_type="issue_rewrite", reason="single occurrence"),
        ],
    )
    productization_path = _write_productization_map(
        tmp_path / "rescue_productization.json",
        [
            {
                "class": "session_restart:runner freshness gate blocked launch",
                "target_kind": "fixture",
                "target": "docs/benchmarks/corpus.json",
                "title": "benchmark corpus",
            }
        ],
    )

    exit_code = mod.main(
        [
            "--path",
            str(ledger.path),
            "--productization-map",
            str(productization_path),
            "--report-json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["summary"]["linked_fixture_count"] == 1
    assert payload["summary"]["one_off_class_count"] == 1
    assert payload["repeated_classes"][0]["productization_status"] == "linked_fixture"
    assert payload["repeated_classes"][0]["productization_target"] == "docs/benchmarks/corpus.json"


def test_load_productization_map_returns_empty_for_missing_file(tmp_path: Path) -> None:
    loaded = mod.load_productization_map(tmp_path / "missing.json")

    assert loaded == {}
