from __future__ import annotations

import json
import sys
from subprocess import CompletedProcess
from pathlib import Path

from aragora.swarm.rescue_events import RescueEvent, RescueEventLedger

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import rescue_to_fixtures as mod  # noqa: E402


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


def test_build_issue_drafts_only_for_unlinked_repeated_classes(tmp_path: Path) -> None:
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
    productization_map_path = _write_productization_map(
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

    report = mod.load_rescue_productization_report(
        ledger_path=ledger.path,
        threshold=2,
        productization_map_path=productization_map_path,
    )
    drafts = mod.build_issue_drafts(report)

    assert len(drafts) == 1
    assert drafts[0]["class"] == "followup_prompt:needs explicit next step from founder"
    assert drafts[0]["issue_numbers"] == [5512, 5515]
    assert "Productize the repeated rescue class" in drafts[0]["body"]


def test_main_json_reports_linked_classes_and_issue_drafts(tmp_path: Path, capsys) -> None:
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
    productization_map_path = _write_productization_map(
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
            str(productization_map_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["summary"]["linked_fixture_count"] == 1
    assert payload["summary"]["unlinked_repeated_class_count"] == 1
    assert len(payload["issue_drafts"]) == 1
    assert (
        payload["issue_drafts"][0]["class"]
        == "followup_prompt:needs explicit next step from founder"
    )


def test_create_substrate_issues_dry_run_uses_issue_drafts_only(tmp_path: Path) -> None:
    ledger = _ledger_with_events(
        tmp_path,
        [
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

    report = mod.load_rescue_productization_report(
        ledger_path=ledger.path,
        threshold=2,
        productization_map_path=tmp_path / "missing.json",
    )
    drafts = mod.build_issue_drafts(report)
    results = mod.create_substrate_issues(drafts, dry_run=True)

    assert len(results) == 1
    assert results[0].startswith("DRY-RUN: would create '[TW-03] Productize repeated rescue class:")


def test_create_substrate_issues_records_issue_link_in_productization_map(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ledger = _ledger_with_events(
        tmp_path,
        [
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
    productization_map_path = tmp_path / "rescue_productization.json"
    report = mod.load_rescue_productization_report(
        ledger_path=ledger.path,
        threshold=2,
        productization_map_path=productization_map_path,
    )
    drafts = mod.build_issue_drafts(report)

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="https://github.com/synaptent/aragora/issues/6001\n",
            stderr="",
        ),
    )

    results = mod.create_substrate_issues(
        drafts,
        repo="synaptent/aragora",
        dry_run=False,
        productization_map_path=productization_map_path,
    )
    refreshed = mod.load_rescue_productization_report(
        ledger_path=ledger.path,
        threshold=2,
        productization_map_path=productization_map_path,
    )

    assert len(results) == 1
    assert "#6001" in results[0]
    assert refreshed["repeated_classes"][0]["productization_status"] == "linked_issue"
    assert refreshed["repeated_classes"][0]["productization_target"] == "#6001"
