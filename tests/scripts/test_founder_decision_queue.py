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


def _single_item_packet(*, generated: str, target: str, reply: str) -> str:
    return f"""# Founder Decision Queue Packet

Generated: {generated}

## Pending Rulings

| Priority | Link | Current blocker | Requested action | One-word reply |
| --- | --- | --- | --- | --- |
| 1 | {target} | Needs a decision. | Rule on this item. | `{reply}` |

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


def test_pending_rulings_parser_stops_before_nested_tables() -> None:
    packet = (
        _single_item_packet(
            generated="2026-07-05T13:07:20Z",
            target="PR #1: https://github.com/synaptent/aragora/pull/1",
            reply="approve",
        )
        + """
### Follow-up Table

| Priority | Link | Current blocker | Requested action | One-word reply |
| --- | --- | --- | --- | --- |
| 2 | PR #2: https://github.com/synaptent/aragora/pull/2 | Not pending. | Ignore. | `release` |
"""
    )

    items = fdq.parse_decision_packet(packet, source="local.md")

    assert len(items) == 1
    assert items[0].target.startswith("PR #1")


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


def test_collect_decision_items_keeps_newest_packet_on_thread(tmp_path: Path) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                reply="approve",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T11:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T11:00:00Z",
                target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                reply="hold",
            ),
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert len(items) == 1
    assert items[0].expected_reply == "hold"


def test_collect_decision_items_omits_ruled_after_packet(tmp_path: Path) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                reply="approve",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:05:00Z",
            "thread_state": "open",
            "body": "approve",
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert items == []


def test_collect_decision_items_requires_exact_one_word_reply_case(tmp_path: Path) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8886: https://github.com/synaptent/aragora/pull/8886",
                reply="B",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:05:00Z",
            "thread_state": "open",
            "body": "b",
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert len(items) == 1
    assert items[0].expected_reply == "B"


def test_collect_decision_items_accepts_exact_one_word_reply(tmp_path: Path) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8886: https://github.com/synaptent/aragora/pull/8886",
                reply="B",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:05:00Z",
            "thread_state": "open",
            "body": "`B`",
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert items == []


def test_collect_decision_items_settlement_match_uses_target_number_boundary(
    tmp_path: Path,
) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #875: https://github.com/synaptent/aragora/pull/875",
                reply="approve",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:05:00Z",
            "thread_state": "open",
            "body": "Settlement recorded for PR #8756.",
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert len(items) == 1
    assert items[0].target.startswith("PR #875:")


def test_collect_decision_items_settlement_match_resolves_exact_target(
    tmp_path: Path,
) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                reply="approve",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:05:00Z",
            "thread_state": "open",
            "body": "Settlement recorded for PR #8756.",
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert items == []


def test_collect_decision_items_omits_closed_thread_packet(tmp_path: Path) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "closed",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                reply="approve",
            ),
        }
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert items == []


def test_collect_decision_items_keeps_open_unruled_packet(tmp_path: Path) -> None:
    comments = [
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:00:00Z",
            "thread_state": "open",
            "body": _single_item_packet(
                generated="2026-07-05T10:00:00Z",
                target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                reply="approve",
            ),
        },
        {
            "html_url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
            "issue_url": "https://api.github.com/repos/synaptent/aragora/issues/8845",
            "created_at": "2026-07-05T10:05:00Z",
            "thread_state": "open",
            "body": "not a ruling",
        },
    ]
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        issue_comments_json=comments_path,
    )

    assert len(items) == 1
    assert items[0].expected_reply == "approve"


def test_github_issue_sources_normalize_live_comment_payload() -> None:
    payload = {
        "state": "OPEN",
        "url": "https://github.com/synaptent/aragora/issues/8845",
        "comments": [
            {
                "url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
                "createdAt": "2026-07-05T10:00:00Z",
                "body": _single_item_packet(
                    generated="2026-07-05T10:00:00Z",
                    target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                    reply="approve",
                ),
            },
            {
                "url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-2",
                "createdAt": "2026-07-05T10:05:00Z",
                "body": "approve",
            },
        ],
    }

    sources = fdq._github_issue_comment_sources(
        repo="synaptent/aragora",
        issue="8845",
        payload=payload,
    )
    items = []
    for source in sources:
        for item in fdq.parse_decision_packet(source.body, source=source.source):
            if not fdq._item_resolved_after_packet(source, item):
                items.append(item)

    assert items == []


def test_collect_decision_items_fetches_github_issue_read_only(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "state": "OPEN",
                "url": "https://github.com/synaptent/aragora/issues/8845",
                "comments": [
                    {
                        "url": "https://github.com/synaptent/aragora/issues/8845#issuecomment-1",
                        "createdAt": "2026-07-05T10:00:00Z",
                        "body": _single_item_packet(
                            generated="2026-07-05T10:00:00Z",
                            target="PR #8756: https://github.com/synaptent/aragora/pull/8756",
                            reply="approve",
                        ),
                    }
                ],
            }
        )

    def fake_run(cmd: list[str], **kwargs: Any) -> Completed:
        calls.append(cmd)
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return Completed()

    monkeypatch.setattr(fdq.subprocess, "run", fake_run)

    items = fdq.collect_decision_items(
        decisions_root=tmp_path / "empty",
        github_issues=["8845"],
        repo="synaptent/aragora",
    )

    assert len(items) == 1
    assert items[0].expected_reply == "approve"
    assert calls == [
        [
            "gh",
            "issue",
            "view",
            "8845",
            "--repo",
            "synaptent/aragora",
            "--json",
            "state,url,comments",
        ]
    ]
