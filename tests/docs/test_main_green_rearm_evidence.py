from pathlib import Path


RUNBOOK = Path("docs/runbooks/main-green-rearm-evidence.md")


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
    assert ".app_id == $requirement.app_id" in text
    assert "| jq '[.[].check_runs[]" in text
    assert 'tee "$EVIDENCE_DIR/required-contexts.json"' in text
    assert '.latest.conclusion == "success"' in text
    assert '.context == "aragora-merge-quorum"' in text
    assert '.latest.conclusion == "skipped"' in text
    assert 'all(.statuses[]; .found and .latest.state == "success")' in text
