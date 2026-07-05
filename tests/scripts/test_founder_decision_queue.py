from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "founder_decision_queue.py"
    spec = importlib.util.spec_from_file_location("founder_decision_queue_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fdq = _load_module()


PACKET = """# Founder Decision Queue Packet

Generated: 2026-07-05T13:07:20Z

## Pending Rulings

| Priority | Link | Current blocker | Requested action | One-word reply |
| --- | --- | --- | --- | --- |
| 1 | PR #8756: https://github.com/synaptent/aragora/pull/8756 | Tier 4 blocked. | Approve exact-head Tier 4 preapproval. | `approve` |
| 2 | PR #8406: https://github.com/synaptent/aragora/pull/8406 | Stale owner. | Release or preserve the stale owner claim. | `release` |
| 3 | PR #8886 decision request: https://github.com/synaptent/aragora/pull/8886#issuecomment-4886125338 | Policy crux. | Choose Option B. | `B` |

## Current Live Snapshots

Not part of the decision table.
"""


def test_parse_decision_packet_extracts_pending_rulings() -> None:
    items = fdq.parse_decision_packet(PACKET, source="local.md")

    assert [item.item for item in items] == ["Priority 1", "Priority 2", "Priority 3"]
    assert items[0].target.startswith("PR #8756")
    assert items[0].expected_reply == "approve"
    assert items[1].expected_reply == "release"
    assert items[2].expected_reply == "B"
    assert items[0].packet_generated_at.isoformat() == "2026-07-05T13:07:20+00:00"


def test_render_markdown_empty_queue() -> None:
    rendered = fdq.render_markdown([], now=fdq._parse_datetime("2026-07-05T14:00:00Z"))

    assert rendered.startswith("# Founder Decision Queue")
    assert "No pending operator rulings found." in rendered
    assert "| Item |" not in rendered


def test_collect_decision_items_deduplicates_issue_comment_export(tmp_path: Path) -> None:
    decisions_root = tmp_path / "founder-decisions"
    decisions_root.mkdir()
    (decisions_root / "packet.md").write_text(PACKET, encoding="utf-8")
    comments = [
        {
            "html_url": "https://example.test/comment",
            "body": PACKET,
        }
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=decisions_root,
        issue_comments_json=comments_path,
    )

    assert len(items) == 3
    rendered = fdq.render_markdown(
        items,
        now=fdq._parse_datetime("2026-07-05T15:07:20Z"),
    )
    assert "| Priority 1 | PR #8756:" in rendered
    assert "`approve`" in rendered
    assert "2.0h" in rendered
