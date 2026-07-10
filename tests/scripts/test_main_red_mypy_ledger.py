from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "main_red_mypy_ledger.py"
SPEC = importlib.util.spec_from_file_location("main_red_mypy_ledger", SCRIPT_PATH)
assert SPEC and SPEC.loader
ledger = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ledger
SPEC.loader.exec_module(ledger)


def test_parse_mypy_output_accepts_both_diagnostic_shapes() -> None:
    findings = ledger.parse_mypy_output(
        "\n".join(
            [
                "aragora/server/a.py:12:4: error: Bad call  [arg-type]",
                "aragora/nomic/b.py:7: error: Missing return  [return]",
                "Found 2 errors in 2 files (checked 4 source files)",
            ]
        )
    )

    assert [finding.path for finding in findings] == [
        "aragora/server/a.py",
        "aragora/nomic/b.py",
    ]
    assert findings[0].column == 4
    assert findings[0].code == "arg-type"
    assert findings[1].column is None
    assert findings[1].code == "return"


def test_parse_claim_comments_ignores_expired_and_template_claims() -> None:
    comments = [
        {
            "body": """CLAIM
owner: active-owner
branch: codex/active
files:
  - aragora/server/a.py
expires_at: 2026-07-11T00:00:00Z
"""
        },
        {
            "body": """CLAIM
owner: expired-owner
files:
  - aragora/nomic/b.py
expires_at: 2026-07-09T00:00:00Z
"""
        },
        {
            "body": """CLAIM
owner: <GitHub login>
files:
  - <exact path>
expires_at: <RFC3339 timestamp>
"""
        },
    ]

    claims = ledger.parse_claim_comments(comments, now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert len(claims) == 1
    assert claims[0].owner == "active-owner"
    assert claims[0].branch == "codex/active"
    assert claims[0].files == frozenset({"aragora/server/a.py"})


def test_reconcile_prefers_pr_files_and_links_claim_branch() -> None:
    claims = [
        ledger.Claim(
            owner="owner-a",
            branch="codex/active",
            files=frozenset({"aragora/server/a.py"}),
            expires_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        ),
        ledger.Claim(
            owner="owner-b",
            branch=None,
            files=frozenset({"aragora/connectors/c.py"}),
            expires_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        ),
    ]
    pull_requests = [
        ledger.PullRequest(
            number=10,
            branch="codex/active",
            state="OPEN",
            files=frozenset(),
        ),
        ledger.PullRequest(
            number=11,
            branch="codex/other",
            state="MERGED",
            files=frozenset({"aragora/nomic/b.py"}),
        ),
    ]

    statuses = ledger.reconcile_file_statuses(claims, pull_requests)

    assert statuses["aragora/server/a.py"] == ledger.WorkStatus("open_pr", "OPEN PR #10")
    assert statuses["aragora/connectors/c.py"] == ledger.WorkStatus("claimed", "CLAIMED owner-b")
    assert statuses["aragora/nomic/b.py"] == ledger.WorkStatus("merged", "MERGED PR #11")


def test_bucket_rows_and_totals_report_partial_coverage() -> None:
    findings = ledger.parse_mypy_output(
        "\n".join(
            [
                "aragora/server/a.py:1:1: error: A  [arg-type]",
                "aragora/server/a.py:2:1: error: B  [arg-type]",
                "aragora/server/unclaimed.py:3: error: C  [return]",
                "aragora/nomic/b.py:4: error: D  [return]",
                "scripts/tool.py:5: error: E  [assignment]",
            ]
        )
    )
    statuses = {
        "aragora/server/a.py": ledger.WorkStatus("open_pr", "OPEN PR #10"),
        "aragora/nomic/b.py": ledger.WorkStatus("claimed", "CLAIMED owner-b"),
    }

    rows = ledger.build_bucket_rows(findings, statuses)
    report = ledger.render_markdown(
        findings,
        rows,
        head_sha="abc123",
        command=ledger.TYPECHECK_COMMAND,
        command_exit=1,
        diagnostic_command=None,
        diagnostic_exit=None,
        gate_false_green=False,
        enforce_requested=False,
    )

    server = next(row for row in rows if row.bucket == "aragora/server")
    assert server.error_count == 3
    assert server.open_pr_error_count == 2
    assert server.status == "UNCLAIMED (2/3 covered: OPEN PR #10)"
    assert "40.0% covered by open PRs (2)" in report
    assert "2 unclaimed buckets" in report


def test_resolve_typecheck_findings_exposes_false_green_gate() -> None:
    result = ledger.TypecheckResult(
        gate_exit=0,
        gate_output="Type check passed (0 errors)\n",
        diagnostic_exit=1,
        diagnostic_output="aragora/server/a.py:12: error: Bad call  [arg-type]\n",
    )

    findings, false_green = ledger.resolve_typecheck_findings(result)

    assert len(findings) == 1
    assert findings[0].path == "aragora/server/a.py"
    assert false_green is True


def test_run_typecheck_falls_back_when_gate_has_no_diagnostics(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, "Type check passed (0 errors)\n", "")
        return subprocess.CompletedProcess(
            command,
            1,
            "aragora/server/a.py:12: error: Bad call  [arg-type]\n",
            "",
        )

    monkeypatch.setattr(ledger.subprocess, "run", fake_run)

    result = ledger.run_typecheck(tmp_path, timeout=30)

    assert calls[0] == ledger.TYPECHECK_COMMAND
    assert calls[1][1:] == ledger.DIAGNOSTIC_COMMAND[1:]
    assert result.gate_exit == 0
    assert result.diagnostic_exit == 1


def test_main_replays_files_and_enforce_stub_stays_advisory(tmp_path: Path, capsys) -> None:
    output = tmp_path / "mypy.txt"
    output.write_text("aragora/server/a.py:12:4: error: Bad call  [arg-type]\n", encoding="utf-8")
    snapshot = tmp_path / "claims.json"
    snapshot.write_text(
        json.dumps(
            {
                "comments": [],
                "pull_requests": [
                    {
                        "number": 10,
                        "headRefName": "codex/active",
                        "state": "OPEN",
                        "files": [{"path": "aragora/server/a.py"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ledger.main(
        [
            "--repo-root",
            str(tmp_path),
            "--claims-json",
            str(snapshot),
            "--typecheck-output",
            str(output),
            "--now",
            "2026-07-10T00:00:00Z",
            "--enforce",
        ]
    )

    captured = capsys.readouterr().out
    assert result == 0
    assert "OPEN PR #10" in captured
    assert "Enforcement: requested but intentionally not implemented" in captured
