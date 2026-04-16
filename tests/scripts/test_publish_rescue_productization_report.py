from __future__ import annotations

import json
import sys
from pathlib import Path

from aragora.swarm.rescue_events import RescueEvent, RescueEventLedger

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import publish_rescue_productization_report as mod  # noqa: E402


def _ledger_with_events(tmp_path: Path, events: list[RescueEvent]) -> RescueEventLedger:
    ledger = RescueEventLedger(path=tmp_path / "rescue_events.jsonl")
    for event in events:
        ledger.record(event)
    return ledger


def test_build_published_report_links_existing_issue_and_updates_map(
    tmp_path: Path, monkeypatch
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
    mod.write_productization_map_payload(
        productization_map_path,
        {
            "schema_version": 1,
            "entries": [],
        },
    )

    monkeypatch.setattr(
        mod,
        "find_existing_issue_by_title",
        lambda **_: {
            "number": 6001,
            "title": "[TW-03] Productize repeated rescue class: followup-prompt-needs-explicit-next-step-from-founder",
            "url": "https://github.com/synaptent/aragora/issues/6001",
            "state": "open",
        },
    )

    payload = mod.build_published_report(
        ledger_path=ledger.path,
        productization_map_path=productization_map_path,
        repo="synaptent/aragora",
        generated_at="2026-04-14T18:35:00Z",
        ensure_issues=True,
    )

    assert payload["summary"]["linked_issue_count"] == 1
    assert payload["issue_drafts"] == []
    assert payload["issue_linkage_results"] == [
        {
            "action": "linked_existing_issue",
            "class": "followup_prompt:needs explicit next step from founder",
            "target": "#6001",
            "target_kind": "issue",
            "url": "https://github.com/synaptent/aragora/issues/6001",
        }
    ]
    written_map = json.loads(productization_map_path.read_text(encoding="utf-8"))
    assert written_map["entries"] == [
        {
            "class": "followup_prompt:needs explicit next step from founder",
            "notes": "Auto-linked by recurring TW-03 harvest.",
            "target": "#6001",
            "target_kind": "issue",
            "title": "[TW-03] Productize repeated rescue class: followup-prompt-needs-explicit-next-step-from-founder",
        }
    ]


def test_publish_report_bundle_writes_timestamped_and_latest(tmp_path: Path) -> None:
    payload = {
        "generated_at": "2026-04-14T18:36:07Z",
        "summary": {"repeated_class_count": 0},
    }

    written = mod.publish_report_bundle(
        publish_dir=tmp_path / "published",
        payload=payload,
    )

    assert written["timestamped"] == (
        tmp_path / "published" / "rescue-productization-20260414T183607Z.json"
    )
    assert written["latest"] == tmp_path / "published" / "latest.json"
    assert json.loads(written["latest"].read_text(encoding="utf-8"))["generated_at"] == (
        "2026-04-14T18:36:07Z"
    )
