import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


RUNBOOK = Path("docs/runbooks/main-green-rearm-evidence.md")


def _reconciliation_program(text: str) -> str:
    match = re.search(
        r"jq -n \\\n.*?\n  '(\{\n.*?\n  \})' \\\n  \| tee",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def test_toolchain_requirement_is_read_from_candidate_commit() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert 'git show "${CANDIDATE_SHA}:pyproject.toml"' in text
    assert "grep -nE '\"mypy[<>=]'" in text
    assert '"$REPO_ROOT/pyproject.toml"' not in text


def test_required_context_reconciliation_consumes_all_check_run_pages() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "gh api --paginate --slurp" in text
    assert "check-runs?filter=latest&per_page=100" in text
    assert "statuses?per_page=100" in text
    assert "checks: [.checks[] | {context, app_id}]" in text
    assert "legacy_contexts: ([.contexts[]] - [.checks[].context] | unique)" in text
    assert "app_id: (.app.id // null)" in text
    assert "created_at," in text
    assert ".app_id == $requirement.app_id" in text
    assert "| jq '[.[].check_runs[]" in text
    assert 'tee "$EVIDENCE_DIR/required-contexts.json"' in text
    assert '.latest.conclusion == "success"' in text
    assert '.context == "aragora-merge-quorum"' in text
    assert '.latest.conclusion == "skipped"' in text
    assert 'all(.statuses[]; .found and .latest.state == "success")' in text


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is required for runbook fixture")
def test_required_context_reconciliation_prefers_newer_queued_run() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    policy = {"checks": [{"context": "lint", "app_id": 15368}], "legacy_contexts": []}
    runs = [
        {
            "id": 100,
            "name": "lint",
            "app_id": 15368,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-07-17T10:00:00Z",
            "started_at": "2026-07-17T10:00:01Z",
        },
        {
            "id": 101,
            "name": "lint",
            "app_id": 15368,
            "status": "queued",
            "conclusion": None,
            "created_at": "2026-07-17T10:01:00Z",
            "started_at": None,
        },
    ]

    result = subprocess.run(
        [
            "jq",
            "-n",
            "--argjson",
            "policy",
            json.dumps(policy),
            "--argjson",
            "runs",
            json.dumps(runs),
            "--argjson",
            "statuses",
            "[]",
            _reconciliation_program(text),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(result.stdout)
    assert evidence["checks"][0]["latest"]["id"] == 101
    assert evidence["checks"][0]["latest"]["status"] == "queued"
