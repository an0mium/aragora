import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


RUNBOOK = Path(__file__).resolve().parents[2] / "docs/runbooks/main-green-rearm-evidence.md"


def _reconciliation_program(text: str) -> str:
    match = re.search(
        r"jq -n \\\n.*?\n  '(\{\n.*?\n  \})' \\\n  \| tee",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _policy_program(text: str) -> str:
    block = text.split('REQUIRED_POLICY_JSON="$(')[1]
    match = re.search(
        r"--argjson protection .*? \\\n    '(\{\n.*?\n    \})'\n\)\" \|\| exit 1",
        block,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _reconciliation_guard(text: str) -> str:
    match = re.search(
        r"jq -e \\\n  '(.*?)' \\\n"
        r'  "\$EVIDENCE_DIR/required-contexts\.json"',
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _rearm_shell_program(text: str) -> str:
    match = re.search(
        r"```bash\nsh -euc '\n(.*?)\n' sh \"\$HALT_FILE\"",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _reconcile(
    text: str,
    *,
    policy: dict[str, object],
    runs: list[dict[str, object]],
    statuses: list[dict[str, object]] | None = None,
) -> dict[str, object]:
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
            json.dumps(statuses or []),
            _reconciliation_program(text),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


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
    assert '"repos/synaptent/aragora/rules/branches/main?per_page=100"' in text
    assert ".parameters.required_status_checks[]?" in text
    assert "repos/synaptent/aragora/branches/main/protection/required_status_checks" in text
    assert 'source: "ruleset"' in text
    assert 'source: "branch_protection"' in text
    assert "status_or_checks:" in text
    assert "- [$ruleset[].context]" in text
    assert "sources:" in text
    assert "app_id: (.app.id // null)" in text
    assert "else ($matches | max_by(.id))" in text
    assert ".app_id == $requirement.app_id" in text
    assert "| jq '[.[].check_runs[]" in text
    assert 'tee "$EVIDENCE_DIR/required-contexts.json"' in text
    assert '.latest.conclusion == "success"' in text
    assert '.context == "aragora-merge-quorum"' in text
    assert '.latest.conclusion == "skipped"' in text
    assert 'all(.statuses[]; .found and .latest.state == "success")' in text


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is required for runbook fixtures")
class TestJqPrograms:
    def test_required_policy_unions_rulesets_and_branch_protection(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        ruleset = [
            {"context": "ruleset-only", "app_id": 15368},
            {"context": "status-or-check", "app_id": None},
        ]
        protection = {
            "checks": [{"context": "lint", "app_id": 15368}],
            "legacy_contexts": ["legacy", "ruleset-only", "status-or-check"],
        }

        result = subprocess.run(
            [
                "jq",
                "-n",
                "--argjson",
                "ruleset",
                json.dumps(ruleset),
                "--argjson",
                "protection",
                json.dumps(protection),
                _policy_program(text),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        policy = json.loads(result.stdout)
        assert policy["checks"] == [
            {
                "context": "lint",
                "app_id": 15368,
                "sources": ["branch_protection"],
            },
            {"context": "ruleset-only", "app_id": 15368, "sources": ["ruleset"]},
        ]
        assert policy["status_or_checks"] == [
            {"context": "status-or-check", "sources": ["ruleset"]}
        ]
        assert policy["legacy_contexts"] == ["legacy"]
        assert policy["sources"] == {"ruleset": True, "branch_protection": True}

    def test_required_context_reconciliation_prefers_newer_queued_run(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [
                {
                    "context": "lint",
                    "app_id": 15368,
                    "sources": ["branch_protection"],
                }
            ],
            "status_or_checks": [],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }
        runs = [
            {
                "id": 100,
                "name": "lint",
                "app_id": 15368,
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-07-17T10:00:01Z",
            },
            {
                "id": 101,
                "name": "lint",
                "app_id": 15368,
                "status": "queued",
                "conclusion": None,
                "started_at": None,
            },
        ]

        evidence = _reconcile(text, policy=policy, runs=runs)
        assert evidence["checks"][0]["latest"]["id"] == 101
        assert evidence["checks"][0]["latest"]["status"] == "queued"

    @pytest.mark.parametrize(
        "runs",
        [
            [],
            [
                {
                    "id": 101,
                    "name": "ruleset-only",
                    "app_id": 15368,
                    "status": "completed",
                    "conclusion": "failure",
                }
            ],
        ],
    )
    def test_ruleset_only_missing_or_red_check_blocks(
        self,
        runs: list[dict[str, object]],
    ) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [
                {
                    "context": "ruleset-only",
                    "app_id": 15368,
                    "sources": ["ruleset"],
                }
            ],
            "status_or_checks": [],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }
        evidence = _reconcile(text, policy=policy, runs=runs)

        result = subprocess.run(
            ["jq", "-e", _reconciliation_guard(text)],
            input=json.dumps(evidence),
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

    def test_unbound_ruleset_context_accepts_green_legacy_status(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [],
            "status_or_checks": [{"context": "ruleset-status", "sources": ["ruleset"]}],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }
        statuses = [
            {
                "id": 7,
                "context": "ruleset-status",
                "state": "success",
                "updated_at": "2026-07-20T12:00:00Z",
            }
        ]

        evidence = _reconcile(text, policy=policy, runs=[], statuses=statuses)
        result = subprocess.run(
            ["jq", "-e", _reconciliation_guard(text)],
            input=json.dumps(evidence),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert evidence["status_or_checks"][0]["latest_status"]["id"] == 7
        assert evidence["status_or_checks"][0]["latest_check"] is None

    def test_unbound_ruleset_context_rejects_conflicting_surfaces(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [],
            "status_or_checks": [{"context": "ruleset-status", "sources": ["ruleset"]}],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }
        runs = [
            {
                "id": 8,
                "name": "ruleset-status",
                "app_id": 15368,
                "status": "completed",
                "conclusion": "success",
            }
        ]
        statuses = [
            {
                "id": 9,
                "context": "ruleset-status",
                "state": "failure",
                "updated_at": "2026-07-20T12:01:00Z",
            }
        ]

        evidence = _reconcile(text, policy=policy, runs=runs, statuses=statuses)
        result = subprocess.run(
            ["jq", "-e", _reconciliation_guard(text)],
            input=json.dumps(evidence),
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert evidence["status_or_checks"][0]["conflict"] is True

    def test_unbound_ruleset_context_rejects_malformed_alternate_proof(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [],
            "status_or_checks": [{"context": "ruleset-status", "sources": ["ruleset"]}],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }
        runs = [
            {
                "id": "not-numeric",
                "name": "ruleset-status",
                "status": "completed",
                "conclusion": "success",
            }
        ]
        statuses = [
            {
                "id": 10,
                "context": "ruleset-status",
                "state": "success",
                "updated_at": "2026-07-20T12:02:00Z",
            }
        ]

        evidence = _reconcile(text, policy=policy, runs=runs, statuses=statuses)
        result = subprocess.run(
            ["jq", "-e", _reconciliation_guard(text)],
            input=json.dumps(evidence),
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert evidence["status_or_checks"][0]["proof_complete"] is False

    def test_app_bound_ruleset_context_rejects_status_only_proof(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [
                {
                    "context": "app-bound",
                    "app_id": 15368,
                    "sources": ["ruleset"],
                }
            ],
            "status_or_checks": [],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }
        statuses = [
            {
                "id": 11,
                "context": "app-bound",
                "state": "success",
                "updated_at": "2026-07-20T12:03:00Z",
            }
        ]

        evidence = _reconcile(text, policy=policy, runs=[], statuses=statuses)
        result = subprocess.run(
            ["jq", "-e", _reconciliation_guard(text)],
            input=json.dumps(evidence),
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert evidence["checks"][0]["found"] is False

    def test_empty_normalized_policy_fails_closed(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        policy = {
            "checks": [],
            "status_or_checks": [],
            "legacy_contexts": [],
            "sources": {"ruleset": True, "branch_protection": True},
        }

        evidence = _reconcile(text, policy=policy, runs=[])
        result = subprocess.run(
            ["jq", "-e", _reconciliation_guard(text)],
            input=json.dumps(evidence),
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert evidence["policy_requirement_count"] == 0


def test_failed_first_rearm_guard_cannot_delete_halt_marker(tmp_path: Path) -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    halt_file = tmp_path / "merge_executor.halt"
    halt_file.write_text("main_red\n", encoding="utf-8")

    result = subprocess.run(
        [
            "sh",
            "-euc",
            _rearm_shell_program(text),
            "sh",
            str(halt_file),
            "definitely-not-the-marker-hash",
            str(tmp_path),
            "deadbeef",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert halt_file.read_text(encoding="utf-8") == "main_red\n"
