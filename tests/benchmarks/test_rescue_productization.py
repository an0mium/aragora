from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import render_rescue_productization_status as mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_rescue_productization_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _section_body(markdown: str, title: str) -> str:
    match = re.search(
        rf"^## {re.escape(title)}\n\n(?P<body>.*?)(?=\n## |\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section: {title}"
    return match.group("body").strip()


def _strip_code(value: str) -> str:
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def _issue_numbers_cell(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "-"
    return ", ".join(f"#{int(item)}" for item in value if isinstance(item, int))


def parse_repeated_class_rows(markdown: str) -> list[dict[str, Any]]:
    body = _section_body(markdown, "Repeated Rescue Classes")
    if body == "- No repeated rescue classes found in the current ledger window.":
        return []

    lines = body.splitlines()
    assert lines[:2] == [
        "| Rescue class | Count | Productization | Target | Example issues |",
        "| --- | --- | --- | --- | --- |",
    ]

    rows: list[dict[str, Any]] = []
    for line in lines[2:]:
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        assert len(columns) == 5, f"unexpected repeated class row: {line}"
        rows.append(
            {
                "class": _strip_code(columns[0]),
                "count": int(columns[1]),
                "productization_status": _strip_code(columns[2]),
                "productization_target": _strip_code(columns[3]),
                "issue_numbers_cell": columns[4],
            }
        )
    return rows


def parse_issue_linkage_actions(markdown: str) -> list[dict[str, str]]:
    body = _section_body(markdown, "Issue Linkage Actions")
    if body == "- none":
        return []

    rows: list[dict[str, str]] = []
    linked_pattern = re.compile(
        r"^- `(?P<action>[^`]+)` `(?P<class>[^`]+)` -> `(?P<target>[^`]+)` "
        r"\(\[link]\((?P<url>[^)]+)\)\)$"
    )
    error_pattern = re.compile(r"^- `(?P<action>[^`]+)` `(?P<class>[^`]+)`: `(?P<error>[^`]+)`$")
    plain_pattern = re.compile(r"^- `(?P<action>[^`]+)` `(?P<class>[^`]+)` -> `(?P<target>[^`]+)`$")

    for line in body.splitlines():
        linked = linked_pattern.match(line)
        if linked is not None:
            rows.append(
                {
                    "action": linked.group("action"),
                    "class": linked.group("class"),
                    "target": linked.group("target"),
                    "url": linked.group("url"),
                    "error": "",
                }
            )
            continue

        errored = error_pattern.match(line)
        if errored is not None:
            rows.append(
                {
                    "action": errored.group("action"),
                    "class": errored.group("class"),
                    "target": "",
                    "url": "",
                    "error": errored.group("error"),
                }
            )
            continue

        plain = plain_pattern.match(line)
        assert plain is not None, f"unexpected issue linkage row: {line}"
        rows.append(
            {
                "action": plain.group("action"),
                "class": plain.group("class"),
                "target": plain.group("target"),
                "url": "",
                "error": "",
            }
        )
    return rows


def parse_issue_drafts(markdown: str) -> list[dict[str, str]]:
    body = _section_body(markdown, "Remaining Issue Drafts")
    if body == "- none":
        return []

    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^- `(?P<class>[^`]+)` -> `(?P<title>[^`]+)`$")
    for line in body.splitlines():
        match = pattern.match(line)
        assert match is not None, f"unexpected issue draft row: {line}"
        rows.append({"class": match.group("class"), "title": match.group("title")})
    return rows


def parse_counted_class_bullets(markdown: str, title: str) -> list[dict[str, Any]]:
    body = _section_body(markdown, title)
    if body == "- none":
        return []

    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^- `(?P<class>[^`]+)` \((?P<count>\d+)x\)$")
    for line in body.splitlines():
        match = pattern.match(line)
        assert match is not None, f"unexpected counted class row in {title}: {line}"
        rows.append({"class": match.group("class"), "count": int(match.group("count"))})
    return rows


def expected_repeated_class_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "class": str(row.get("class") or "").strip(),
            "count": int(row.get("count", 0) or 0),
            "productization_status": (
                str(row.get("productization_status") or "unlinked").strip() or "unlinked"
            ),
            "productization_target": str(row.get("productization_target") or "-").strip() or "-",
            "issue_numbers_cell": _issue_numbers_cell(row.get("issue_numbers")),
        }
        for row in rows
    ]


def expected_issue_linkage_actions(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    expected: list[dict[str, str]] = []
    for row in rows:
        target = str(row.get("target") or "").strip()
        url = str(row.get("url") or "").strip()
        error = str(row.get("error") or "").strip()
        rendered_target = ""
        if not error:
            rendered_target = target if url else (target or "n/a")
        expected.append(
            {
                "action": str(row.get("action") or "").strip(),
                "class": str(row.get("class") or "").strip(),
                "target": rendered_target,
                "url": url,
                "error": error,
            }
        )
    return expected


def expected_issue_drafts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "class": str(row.get("class") or "").strip(),
            "title": str(row.get("title") or "").strip(),
        }
        for row in rows
    ]


def expected_counted_class_bullets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "class": str(row.get("class") or "").strip(),
            "count": int(row.get("count", 0) or 0),
        }
        for row in rows
    ]


def test_render_status_markdown_round_trips_non_empty_payload() -> None:
    report_path = REPO_ROOT / "docs/status/generated/rescue_productization/latest.json"
    payload = {
        "generated_at": "2026-04-17T20:15:00Z",
        "ledger_path": "/Users/test/.aragora/rescue_events.jsonl",
        "productization_map_path": "docs/benchmarks/rescue_productization.json",
        "summary": {
            "repeated_class_count": 2,
            "linked_fixture_count": 0,
            "linked_issue_count": 1,
            "linked_other_count": 0,
            "unlinked_repeated_class_count": 1,
            "one_off_class_count": 1,
            "below_threshold_class_count": 1,
        },
        "repeated_classes": [
            {
                "class": "followup_prompt:needs explicit next step from founder",
                "count": 2,
                "productization_status": "linked_issue",
                "productization_target": "#6001",
                "issue_numbers": [5512, 5515],
            },
            {
                "class": "manual_merge:required review gate",
                "count": 2,
                "productization_status": "unlinked",
                "productization_target": "",
                "issue_numbers": [5617],
            },
        ],
        "issue_linkage_results": [
            {
                "class": "followup_prompt:needs explicit next step from founder",
                "action": "linked_existing_issue",
                "target": "#6001",
                "url": "https://github.com/synaptent/aragora/issues/6001",
            },
            {
                "class": "manual_merge:required review gate",
                "action": "issue_creation_failed",
                "error": "api.github.com unavailable",
            },
        ],
        "issue_drafts": [
            {
                "class": "manual_merge:required review gate",
                "title": "[TW-03] Productize repeated rescue class: manual-merge-required-review-gate",
            }
        ],
        "one_off_classes": [
            {"class": "issue_rewrite:scope contradicted itself", "count": 1},
        ],
        "below_threshold_classes": [
            {"class": "missing_fixture:single occurrence", "count": 1},
        ],
    }

    markdown = mod.render_status_markdown(report_path=report_path, payload=payload)

    assert parse_repeated_class_rows(markdown) == expected_repeated_class_rows(
        payload["repeated_classes"]
    )
    assert parse_issue_linkage_actions(markdown) == expected_issue_linkage_actions(
        payload["issue_linkage_results"]
    )
    assert parse_issue_drafts(markdown) == expected_issue_drafts(payload["issue_drafts"])
    assert parse_counted_class_bullets(markdown, "One-Off Rescue Classes") == (
        expected_counted_class_bullets(payload["one_off_classes"])
    )
    assert parse_counted_class_bullets(markdown, "Below-Threshold Rescue Classes") == (
        expected_counted_class_bullets(payload["below_threshold_classes"])
    )
