import io
import json
import os
import re
import subprocess
import zipfile
from datetime import datetime as RealDateTime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.verify_contract_drift_workflow_state import (
    CANONICAL_WORKFLOW_ID,
    CANONICAL_WORKFLOW_NAME,
    CANONICAL_WORKFLOW_PATH,
    EXPECTED_PROTECTION_STRICT,
    GhApiReader,
    PRE_CUTOVER_REQUIRED_CHECKS,
    VerificationError,
    verify_workflow_state,
)

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github/workflows/contract-drift-governance.yml").read_text()
DOC = yaml.load(TEXT, Loader=yaml.BaseLoader)
JOBS = DOC["jobs"]

LIVE_CHECK_NAMES = (
    "contract-drift-pr-delta",
    "contract-drift-main-receipt",
    "contract-drift-program-trajectory",
)
SOURCE_SHA_EXPR = "${{ github.event_name == 'push' && github.event.after || github.sha }}"
# GitHub Actions app installation id: required-check tuples produced by
# workflow jobs must carry this app identity, never a third-party app.
EXPECTED_PR_DELTA_APP_ID = 15368
GITHUB_RUNNER_ENV_ALLOWLIST = frozenset(
    {
        "CI",
        "COMSPEC",
        "GITHUB_ACTIONS",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "PATHEXT",
        "RUNNER_ARCH",
        "RUNNER_OS",
        "RUNNER_TEMP",
        "RUNNER_TOOL_CACHE",
        "SHELL",
        "SYSTEMROOT",
        "TMPDIR",
        "TZ",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
    }
)


def _analyzer_step(job_id: str) -> dict:
    for step in JOBS[job_id]["steps"]:
        if "run" in step and "check_contract_drift_ratchet.py" in step["run"]:
            return step
    raise AssertionError(f"no analyzer step in {job_id}")


def _upload_step(job_id: str) -> dict:
    for step in JOBS[job_id]["steps"]:
        if str(step.get("uses", "")).startswith("actions/upload-artifact@"):
            return step
    raise AssertionError(f"no upload step in {job_id}")


def _terminal_step() -> dict:
    matches = [
        step
        for step in JOBS["pr-delta"]["steps"]
        if step.get("if") == "always()" and "steps.admission.outcome" in step.get("run", "")
    ]
    assert len(matches) == 1, f"expected one terminal pr-delta step, found {len(matches)}"
    return matches[0]


def _simulate_step(
    run_block: str,
    *,
    env: dict[str, str],
    cwd: Path,
    stubs: dict[str, int] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a workflow run block the way GitHub does (bash -e), with
    optional shell-function stubs standing in for named commands."""
    prelude = "".join(
        f"{name}() {{ echo stub-{name}; return {code}; }}\n" for name, code in (stubs or {}).items()
    )
    runner_env = {
        name: value
        for name in GITHUB_RUNNER_ENV_ALLOWLIST
        if (value := os.environ.get(name)) is not None
    }
    return subprocess.run(
        ["bash", "-e", "-c", prelude + run_block],
        capture_output=True,
        text=True,
        env={**runner_env, **env},
        cwd=str(cwd),
    )


def test_terminal_step_is_selected_by_identity(monkeypatch: pytest.MonkeyPatch):
    terminal = _terminal_step()
    sentinel = {"name": "unrelated trailing cleanup", "run": "exit 99"}
    monkeypatch.setitem(
        JOBS,
        "pr-delta",
        {**JOBS["pr-delta"], "steps": [*JOBS["pr-delta"]["steps"], sentinel]},
    )
    assert _terminal_step() == terminal


def test_simulate_step_scrubs_non_runner_parent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CDG_UNTRUSTED_PARENT_VALUE", "must-not-leak")
    result = _simulate_step(
        'test -z "${CDG_UNTRUSTED_PARENT_VALUE+x}"',
        env={},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def _terminal_repo(tmp_path: Path, *, baseline_text: str) -> tuple[Path, str]:
    """Build a fixture git repo whose BASE_SHA-addressed baseline the terminal
    aggregator inspects, and return (workdir, base_sha)."""
    repo = tmp_path / "terminal-repo"
    baseline = repo / "scripts/baselines/contract_drift_inventory.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(baseline_text, encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c",
            "user.email=cdg-test@example.invalid",
            "-c",
            "user.name=cdg-test",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
    ):
        subprocess.run(argv, cwd=repo, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, base_sha


def _run_terminal(
    tmp_path: Path,
    *,
    outcome: str,
    payload: str | None,
    baseline_text: str = '{"accepted_authority": {}}\n',
) -> subprocess.CompletedProcess[str]:
    repo, base_sha = _terminal_repo(tmp_path, baseline_text=baseline_text)
    if payload is not None:
        (repo / "contract-drift-pr-delta.json").write_text(payload, encoding="utf-8")
    script = _terminal_step()["run"].replace("${{ steps.admission.outcome }}", outcome)
    return _simulate_step(script, env={"BASE_SHA": base_sha}, cwd=repo)


def test_workflow_has_exact_live_checks_and_events():
    assert {job["name"] for job in JOBS.values()} == {f"contract-drift-{name}" for name in ("pr-delta", "main-receipt", "program-trajectory")}  # fmt: skip
    assert set(DOC["on"]) == {"pull_request", "push", "schedule", "workflow_dispatch"} and not {"paths", "paths-ignore"} & set(DOC["on"]["pull_request"])  # fmt: skip
    assert all({"uses": "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", "with": {"python-version": "3.11"}} in job["steps"] for job in JOBS.values())  # fmt: skip


def test_pr_admission_is_event_bound_absolute_and_terminal():
    assert '--base-ref "$BASE_SHA"' in TEXT and '--head-ref "$HEAD_SHA"' in TEXT
    assert TEXT.count('--repo-root "$GITHUB_WORKSPACE"') == 3
    pr = str(JOBS["pr-delta"])
    assert "--mode pr" in pr and "--mode receipt" not in pr and "--mode program" not in pr
    terminal = JOBS["pr-delta"]["steps"][-1]
    assert terminal["if"] == "always()" and "steps.admission.outcome" in terminal["run"]
    receipt, program = JOBS["main-receipt"], JOBS["program-trajectory"]
    assert receipt["env"]["SOURCE_SHA"] == program["env"]["SOURCE_SHA"]
    assert "needs" not in receipt and "needs" not in program
    assert "--mode program" in str(program) and "continue-on-error" not in str(program)


def test_program_trajectory_upload_is_sha_qualified_and_never_masks_the_analyzer():
    program = JOBS["program-trajectory"]
    assert "continue-on-error" not in program  # job level: analyzer red stays red
    *_, analyzer, upload = program["steps"]
    # The analyzer stays the unconditioned terminal enforcement step: no `if`
    # and no `continue-on-error`, so its exit code alone decides the job.
    assert "--mode program" in analyzer["run"]
    assert "tee contract-drift-program-trajectory.json" in analyzer["run"]
    assert "if" not in analyzer and "continue-on-error" not in analyzer
    # The durable upload is exactly one SHA-qualified actions/upload-artifact@v4
    # step that runs on success or failure but not cancellation, and carries no
    # extra keys that could soften or reinterpret the analyzer outcome.
    assert upload == {
        "if": "always() && !cancelled()",
        "uses": "actions/upload-artifact@v4",
        "with": {
            "name": "contract-drift-program-trajectory-${{ github.sha }}",
            "path": "contract-drift-program-trajectory.json",
        },
    }


# --- VAL-CDG-015: live workflow topology -----------------------------------


def _job_names(doc: dict) -> list[str]:
    return [
        str(job.get("name", job_id))
        for job_id, job in doc.get("jobs", {}).items()
        if isinstance(job, dict)
    ]


FIXTURE_REPO = "synaptent/aragora"
FIXTURE_MAIN_SHA = "a" * 40
FIXTURE_RUN_ID = 7001


class FixtureApiReader:
    def __init__(
        self, json_responses: dict[str, list[dict[str, Any]]], bytes_responses: dict[str, bytes]
    ):
        self.json_responses = {key: list(value) for key, value in json_responses.items()}
        self.bytes_responses = bytes_responses
        self.json_calls: list[str] = []

    def get_json(self, endpoint: str) -> dict[str, Any]:
        self.json_calls.append(endpoint)
        responses = self.json_responses.get(endpoint)
        if not responses:
            raise AssertionError(f"unexpected JSON endpoint: {endpoint}")
        return responses.pop(0) if len(responses) > 1 else responses[0]

    def get_bytes(self, endpoint: str, *, max_bytes: int | None = None) -> bytes:
        try:
            payload = self.bytes_responses[endpoint]
        except KeyError as exc:
            raise AssertionError(f"unexpected bytes endpoint: {endpoint}") from exc
        if max_bytes is not None and len(payload) > max_bytes:
            raise VerificationError(f"authenticated GitHub response is too large: {endpoint}")
        return payload


class FixedDateTime(RealDateTime):
    @classmethod
    def now(cls, tz=None):
        return RealDateTime(2026, 2, 19, tzinfo=tz)


def _json_page(items: list[dict[str, Any]], key: str, total: int | None = None) -> dict[str, Any]:
    return {"total_count": len(items) if total is None else total, key: items}


def _artifact_zip(filename: str, payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, json.dumps(payload, sort_keys=True))
    return buffer.getvalue()


def _workflow_record(
    *, workflow_id: int = CANONICAL_WORKFLOW_ID, state: str = "active"
) -> dict[str, Any]:
    return {"id": workflow_id, "name": CANONICAL_WORKFLOW_NAME, "path": CANONICAL_WORKFLOW_PATH, "state": state}  # fmt: skip


def _run_record(
    *,
    run_id: int = FIXTURE_RUN_ID,
    main_sha: str = FIXTURE_MAIN_SHA,
    run_attempt: int = 1,
    started_at: str = "2026-08-12T01:00:00Z",
) -> dict[str, Any]:
    return {"id": run_id, "workflow_id": CANONICAL_WORKFLOW_ID, "path": CANONICAL_WORKFLOW_PATH, "head_branch": "main", "head_sha": main_sha, "event": "push", "run_attempt": run_attempt, "run_started_at": started_at, "status": "completed", "conclusion": "failure"}  # fmt: skip


def _job_record(
    name: str, *, run_id: int, main_sha: str, attempt: int, job_id: int
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    check_id = job_id + 10_000
    check_url = f"https://api.github.com/repos/{FIXTURE_REPO}/check-runs/{check_id}"
    conclusion = dict(zip(LIVE_CHECK_NAMES, ("skipped", "success", "failure"), strict=True))[name]
    job = {"id": job_id, "name": name, "run_attempt": attempt, "check_run_url": check_url}
    check = {"id": check_id, "name": name, "head_sha": main_sha, "app": {"id": EXPECTED_PR_DELTA_APP_ID}, "status": "completed", "conclusion": conclusion, "details_url": f"https://github.com/{FIXTURE_REPO}/actions/runs/{run_id}/job/{job_id}"}  # fmt: skip
    return job, check_url, check


def _live_fixture(
    *,
    workflows: list[dict[str, Any]] | None = None,
    workflow_total: int | None = None,
    runs: list[dict[str, Any]] | None = None,
    run_total: int | None = None,
    run_attempt: int = 1,
    second_snapshot_run: dict[str, Any] | None = None,
    protection_checks: list[dict[str, Any]] | None = None,
    protection_strict: bool = EXPECTED_PROTECTION_STRICT,
    receipt_payload_status: str = "pass",
) -> tuple[FixtureApiReader, dict[str, Any]]:
    run_records = list(runs or [_run_record(run_attempt=run_attempt)])
    selected = max(
        (item for item in run_records if item["head_sha"] == FIXTURE_MAIN_SHA),
        key=lambda item: (item["run_started_at"], item["id"], item["run_attempt"]),
    )
    selected_id = selected["id"]
    selected_sha = selected["head_sha"]
    selected_attempt = selected["run_attempt"]
    workflows = [_workflow_record()] if workflows is None else workflows
    if protection_checks is None:
        protection_checks = [{"context": context, "app_id": app_id} for context, app_id in PRE_CUTOVER_REQUIRED_CHECKS]  # fmt: skip

    json_responses: dict[str, list[dict[str, Any]]] = {
        f"repos/{FIXTURE_REPO}/actions/workflows?per_page=100&page=1": [
            _json_page(workflows, "workflows", workflow_total)
        ],
        f"repos/{FIXTURE_REPO}/git/ref/heads/main": [
            {"object": {"sha": FIXTURE_MAIN_SHA}},
            {"object": {"sha": FIXTURE_MAIN_SHA}},
        ],
        f"repos/{FIXTURE_REPO}/actions/workflows/{CANONICAL_WORKFLOW_ID}/runs?per_page=100&page=1": [
            _json_page(run_records, "workflow_runs", run_total),
            _json_page(run_records, "workflow_runs", run_total),
            _json_page([second_snapshot_run or selected], "workflow_runs"),
            _json_page([second_snapshot_run or selected], "workflow_runs"),
        ],  # fmt: skip
        f"repos/{FIXTURE_REPO}/branches/main/protection/required_status_checks": [
            {"strict": protection_strict, "checks": protection_checks}
        ],
    }
    workflow_source = (ROOT / CANONICAL_WORKFLOW_PATH).read_bytes()
    bytes_responses: dict[str, bytes] = {
        f"repos/{FIXTURE_REPO}/contents/{CANONICAL_WORKFLOW_PATH}?ref=main": workflow_source,
        f"repos/{FIXTURE_REPO}/contents/{CANONICAL_WORKFLOW_PATH}?ref={selected_sha}": workflow_source,
    }

    for attempt in range(1, selected_attempt + 1):
        jobs: list[dict[str, Any]] = []
        for offset, name in enumerate(LIVE_CHECK_NAMES, start=1):
            job, check_url, check = _job_record(
                name,
                run_id=selected_id,
                main_sha=selected_sha,
                attempt=attempt,
                job_id=attempt * 100 + offset,
            )
            jobs.append(job)
            json_responses[check_url] = [check]
        endpoint = f"repos/{FIXTURE_REPO}/actions/runs/{selected_id}/attempts/{attempt}/jobs?per_page=100&page=1"
        json_responses[endpoint] = [_json_page(jobs, "jobs")]

    artifacts: list[dict[str, Any]] = []
    artifact_specs = (
        (9001, "contract-drift-main-receipt", "contract-drift-main-receipt.json", {"source_sha": selected_sha, "status": receipt_payload_status}),
        (9002, "contract-drift-program-trajectory", "contract-drift-program-trajectory.json", {"program": {"source_sha": selected_sha}, "status": "fail"}),
    )  # fmt: skip
    for artifact_id, prefix, filename, payload in artifact_specs:
        raw_zip = _artifact_zip(filename, payload)
        artifacts.append({"id": artifact_id, "name": f"{prefix}-{selected_sha}", "expired": False, "size_in_bytes": len(raw_zip), "workflow_run": {"id": selected_id, "head_sha": selected_sha}})  # fmt: skip
        bytes_responses[f"repos/{FIXTURE_REPO}/actions/artifacts/{artifact_id}/zip"] = raw_zip
    json_responses[
        f"repos/{FIXTURE_REPO}/actions/runs/{selected_id}/artifacts?per_page=100&page=1"
    ] = [_json_page(artifacts, "artifacts")]
    expected = {"workflow_id": CANONICAL_WORKFLOW_ID, "main_sha": FIXTURE_MAIN_SHA, "run_id": selected_id, "run_attempt": selected_attempt}  # fmt: skip
    return FixtureApiReader(json_responses, bytes_responses), expected


def test_exactly_one_active_contract_drift_workflow():
    disabled = [
        {
            "id": 100_000 + index,
            "name": f"legacy-{index}",
            "path": f".github/workflows/legacy-{index}.yml",
            "state": "disabled_manually",
        }
        for index in range(100)
    ]
    workflows = [*disabled, _workflow_record()]
    reader, expected = _live_fixture(workflows=workflows)
    repo = "synaptent/aragora"
    # Force a real second page while preserving one stable response for every later endpoint.
    reader.json_responses[f"repos/{repo}/actions/workflows?per_page=100&page=1"] = [
        _json_page(disabled, "workflows", 101)
    ]
    reader.json_responses[f"repos/{repo}/actions/workflows?per_page=100&page=2"] = [
        _json_page([_workflow_record()], "workflows", 101)
    ]
    result = verify_workflow_state(reader)
    assert result["workflow"] == _workflow_record(workflow_id=expected["workflow_id"])
    assert result["workflow_pages"] == [
        f"repos/{repo}/actions/workflows?per_page=100&page=1",
        f"repos/{repo}/actions/workflows?per_page=100&page=2",
    ]
    reader, _ = _live_fixture(workflows=[_workflow_record(state="disabled_manually")])
    with pytest.raises(VerificationError, match="must be active"):
        verify_workflow_state(reader)


def test_exact_three_live_cdg_check_names():
    reader, _ = _live_fixture()
    result = verify_workflow_state(reader)
    assert set(result["selection"]["selected_attempt"]["checks"]) == set(LIVE_CHECK_NAMES)


def test_no_duplicate_live_cdg_check_names():
    duplicate = _workflow_record(workflow_id=CANONICAL_WORKFLOW_ID + 1)
    duplicate["path"] = ".github/workflows/contract-drift-copy.yml"
    reader, _ = _live_fixture(workflows=[_workflow_record(), duplicate])
    with pytest.raises(VerificationError, match="exactly one registered workflow"):
        verify_workflow_state(reader)
    reader, _ = _live_fixture(workflows=[_workflow_record(), _workflow_record()])
    with pytest.raises(VerificationError, match="duplicate identities"):
        verify_workflow_state(reader)
    reader, _ = _live_fixture(workflow_total=2)
    with pytest.raises(VerificationError, match="do not reconcile to total_count"):
        verify_workflow_state(reader)


def test_contract_drift_triggers_pr_main_schedule_dispatch():
    reader, _ = _live_fixture()
    result = verify_workflow_state(reader)
    assert result["workflow"]["state"] == "active"
    assert set(DOC["on"]) == {"pull_request", "push", "schedule", "workflow_dispatch"}


def test_contract_drift_pr_has_no_governed_path_filter_gap():
    reader, _ = _live_fixture()
    verify_workflow_state(reader)
    for event in ("pull_request", "push"):
        assert not {"paths", "paths-ignore"} & set(DOC["on"][event])


def test_contract_drift_pr_uses_event_bound_full_shas():
    reader, expected = _live_fixture()
    result = verify_workflow_state(reader)
    assert result["main_sha"] == expected["main_sha"]
    env = JOBS["pr-delta"]["env"]
    assert env == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }


def test_contract_drift_non_pr_events_resolve_one_sha_for_receipt_and_program():
    reader, expected = _live_fixture()
    result = verify_workflow_state(reader)
    assert result["selection"]["identity"]["run_id"] == expected["run_id"]
    assert (
        JOBS["main-receipt"]["env"]["SOURCE_SHA"] == JOBS["program-trajectory"]["env"]["SOURCE_SHA"]
    )


def _analyzer_step_text(job: dict) -> str:
    for step in job["steps"]:
        if "run" in step and "check_contract_drift_ratchet.py" in step["run"]:
            return step["run"]
    raise AssertionError("no analyzer step")


def test_workflow_runs_paginate_unfiltered_and_filter_locally():
    newer_other_sha = _run_record(
        run_id=8000,
        main_sha="b" * 40,
        started_at="2026-08-12T02:00:00Z",
    )
    reader, expected = _live_fixture(runs=[_run_record(), newer_other_sha])
    result = verify_workflow_state(reader)
    assert result["selection"]["identity"]["run_id"] == expected["run_id"] == 7001
    first_run_endpoint = f"repos/synaptent/aragora/actions/workflows/{CANONICAL_WORKFLOW_ID}/runs?per_page=100&page=1"
    assert first_run_endpoint in reader.json_calls
    assert not any(
        token in first_run_endpoint for token in ("branch=", "status=", "event=", "conclusion=")
    )


def test_filtered_history_requires_disjoint_date_shards_below_1000(monkeypatch: pytest.MonkeyPatch):
    import scripts.verify_contract_drift_workflow_state as verifier

    counts = {
        verifier.HISTORY_START_DATE + verifier.timedelta(days=offset): 400 for offset in range(7)
    }
    monkeypatch.setattr(verifier, "datetime", FixedDateTime)
    calls: list[verifier.date] = []

    def fake_day_runs(reader, *, repo: str, workflow_id: int, day: verifier.date):
        calls.append(day)
        records = [{"id": len(calls) * 1000 + index} for index in range(counts[day])]
        return records, [f"created={day.isoformat()}"]

    monkeypatch.setattr(verifier, "_day_runs", fake_day_runs)
    records, endpoints = verifier._workflow_runs_by_date(
        object(),
        repo="synaptent/aragora",
        workflow_id=CANONICAL_WORKFLOW_ID,
        reported_total=2800,
    )
    logical_shards = [endpoint for endpoint in endpoints if ".." in endpoint]
    assert len(records) == 2800
    assert logical_shards == ["created=2026-02-13..2026-02-14", "created=2026-02-15..2026-02-16", "created=2026-02-17..2026-02-18", "created=2026-02-19..2026-02-19"]  # fmt: skip


def test_workflow_run_ids_reconcile_to_total_count():
    reader, _ = _live_fixture(run_total=2)
    with pytest.raises(VerificationError, match="do not reconcile to total_count"):
        verify_workflow_state(reader)
    duplicate = _run_record(run_id=7001)
    reader, _ = _live_fixture(runs=[duplicate, dict(duplicate)], run_total=2)
    with pytest.raises(VerificationError, match="duplicate identities"):
        verify_workflow_state(reader)


def test_attempts_are_enumerated_one_through_run_attempt():
    reader, expected = _live_fixture(run_attempt=3)
    result = verify_workflow_state(reader)
    assert [attempt["attempt"] for attempt in result["selection"]["attempts"]] == [1, 2, 3]
    assert result["selection"]["identity"]["run_attempt"] == expected["run_attempt"]


def test_jobs_and_check_urls_are_attempt_specific():
    reader, expected = _live_fixture(run_attempt=2)
    result = verify_workflow_state(reader)
    for attempt in result["selection"]["attempts"]:
        assert f"/attempts/{attempt['attempt']}/jobs" in attempt["jobs_endpoint"]
        for check in attempt["checks"].values():
            assert check["jobs_endpoint"] == attempt["jobs_endpoint"]
            assert check["check_url"].startswith(
                "https://api.github.com/repos/synaptent/aragora/check-runs/"
            )
    hostile, _ = _live_fixture(run_attempt=1)
    endpoint = f"repos/synaptent/aragora/actions/runs/{expected['run_id']}/attempts/1/jobs?per_page=100&page=1"
    hostile.json_responses[endpoint][0]["jobs"][0]["run_attempt"] = 2
    with pytest.raises(VerificationError, match="job record is not attempt-specific"):
        verify_workflow_state(hostile)
    hostile, _ = _live_fixture(run_attempt=1)
    hostile.json_responses[endpoint][0]["jobs"][0]["check_run_url"] = (
        "https://api.github.com/repos/attacker/other/check-runs/10101"
    )
    with pytest.raises(VerificationError, match="bound to another repository"):
        verify_workflow_state(hostile)


def test_run_level_artifacts_require_payload_and_release_binding():
    reader, expected = _live_fixture()
    result = verify_workflow_state(reader)
    assert {item["payload_status"] for item in result["selection"]["artifacts"]} == {"pass", "fail"}
    assert all(
        f"-{expected['main_sha']}" in item["name"] for item in result["selection"]["artifacts"]
    )
    hostile, _ = _live_fixture(receipt_payload_status="fail")
    with pytest.raises(VerificationError, match="payload status contradicts"):
        verify_workflow_state(hostile)
    hostile, _ = _live_fixture()
    endpoint = (
        f"repos/synaptent/aragora/actions/runs/{expected['run_id']}/artifacts?per_page=100&page=1"
    )
    hostile.json_responses[endpoint][0]["artifacts"][0]["workflow_run"]["id"] += 1
    with pytest.raises(VerificationError, match="bound to another run or SHA"):
        verify_workflow_state(hostile)


def test_pr_delta_cutover_preserves_exact_context_app_id_set_and_strict():
    after = [
        {"context": context, "app_id": app_id} for context, app_id in PRE_CUTOVER_REQUIRED_CHECKS
    ] + [{"context": "contract-drift-pr-delta", "app_id": EXPECTED_PR_DELTA_APP_ID}]
    reader, _ = _live_fixture(protection_checks=after)
    result = verify_workflow_state(reader)
    assert result["branch_protection"]["cutover_phase"] == "after"
    strict_hostile, _ = _live_fixture(protection_strict=not EXPECTED_PROTECTION_STRICT)
    with pytest.raises(VerificationError, match="strict moved"):
        verify_workflow_state(strict_hostile)


def test_pr_delta_uses_expected_app_identity():
    hostile_checks = [
        {"context": context, "app_id": app_id} for context, app_id in PRE_CUTOVER_REQUIRED_CHECKS
    ] + [{"context": "contract-drift-pr-delta", "app_id": 99999}]
    reader, _ = _live_fixture(protection_checks=hostile_checks)
    with pytest.raises(VerificationError, match="valid cutover state"):
        verify_workflow_state(reader)
    reader, _ = _live_fixture()
    check_url = "https://api.github.com/repos/synaptent/aragora/check-runs/10101"
    reader.json_responses[check_url][0]["app"]["id"] = 99999
    with pytest.raises(VerificationError, match="not authenticated by the GitHub Actions app"):
        verify_workflow_state(reader)


def test_live_verifier_rejects_main_or_selected_run_movement():
    reader, _ = _live_fixture()
    reader.json_responses["repos/synaptent/aragora/git/ref/heads/main"][-1] = {
        "object": {"sha": "b" * 40}
    }
    with pytest.raises(VerificationError, match="matches current main|moved during verification"):
        verify_workflow_state(reader)
    moved = _run_record(run_id=7002, started_at="2026-08-12T02:00:00Z")
    reader, _ = _live_fixture(second_snapshot_run=moved)
    with pytest.raises(VerificationError, match="moved during verification"):
        verify_workflow_state(reader)


def test_terminal_aggregator_fails_if_pr_delta_is_skipped_cancelled_or_missing(tmp_path):
    assert _terminal_step()["if"] == "always()"
    payload = '{"admitted": true}\n'
    assert _run_terminal(tmp_path / "ok", outcome="success", payload=payload).returncode == 0
    for index, outcome in enumerate(("skipped", "cancelled", "failure")):
        result = _run_terminal(tmp_path / f"bad{index}", outcome=outcome, payload=payload)
        assert result.returncode != 0, outcome
    assert _run_terminal(tmp_path / "missing", outcome="success", payload=None).returncode != 0


def test_main_receipt_job_is_distinct_and_successful_when_trajectory_is_red(tmp_path):
    receipt, program = JOBS["main-receipt"], JOBS["program-trajectory"]
    assert receipt["name"] != program["name"] and "needs" not in receipt
    env = {"SOURCE_SHA": "b" * 40, "GITHUB_WORKSPACE": str(tmp_path)}
    red = _simulate_step(_analyzer_step_text(program), env=env, cwd=tmp_path, stubs={"python3": 17})
    green = _simulate_step(
        _analyzer_step_text(receipt), env=env, cwd=tmp_path, stubs={"python3": 0}
    )
    assert red.returncode == 17 and green.returncode == 0
    # The receipt artifact is gated only on the receipt job's own success.
    assert _upload_step("main-receipt")["if"] == "success()"


def test_main_receipt_requires_complete_first_parent_backfill(tmp_path):
    receipt = _analyzer_step_text(JOBS["main-receipt"])
    assert "--mode receipt" in receipt and "--mode program" not in receipt
    env = {"SOURCE_SHA": "b" * 40, "GITHUB_WORKSPACE": str(tmp_path)}
    # An incomplete first-parent backfill is a red receipt analyzer; pipefail
    # preserves that exit through tee, and the success-gated upload withholds
    # the receipt artifact.
    incomplete = _simulate_step(receipt, env=env, cwd=tmp_path, stubs={"python3": 3})
    assert incomplete.returncode == 3
    assert _upload_step("main-receipt")["if"] == "success()"
    assert "continue-on-error" not in str(JOBS["main-receipt"])


def test_program_trajectory_preserves_real_red_exit(tmp_path):
    program = JOBS["program-trajectory"]
    assert "continue-on-error" not in str(program)
    analyzer = _analyzer_step(program_id := "program-trajectory")
    assert "if" not in analyzer, program_id
    env = {"SOURCE_SHA": "c" * 40, "GITHUB_WORKSPACE": str(tmp_path)}
    result = _simulate_step(
        _analyzer_step_text(program), env=env, cwd=tmp_path, stubs={"python3": 7}
    )
    assert result.returncode == 7
    assert (tmp_path / "contract-drift-program-trajectory.json").exists()


def test_contract_drift_authoritative_steps_propagate_nonzero_exit(tmp_path):
    env = {
        "BASE_SHA": "a" * 40,
        "HEAD_SHA": "b" * 40,
        "SOURCE_SHA": "c" * 40,
        "GITHUB_WORKSPACE": str(tmp_path),
    }
    for job_id in ("pr-delta", "main-receipt", "program-trajectory"):
        workdir = tmp_path / job_id
        workdir.mkdir()
        result = _simulate_step(
            _analyzer_step(job_id)["run"],
            env=env,
            cwd=workdir,
            stubs={"python3": 5, "git": 0, "date": 0},
        )
        assert result.returncode == 5, (job_id, result.stderr)


def test_every_contract_drift_authoritative_pipeline_enables_pipefail_before_first_pipe(tmp_path):
    pipe = re.compile(r"(?<!\|)\|(?!\|)")
    for job_id in ("pr-delta", "main-receipt", "program-trajectory"):
        run = _analyzer_step(job_id)["run"]
        lines = run.splitlines()
        first_pipe = next(index for index, line in enumerate(lines) if pipe.search(line))
        assert "set -euo pipefail" in lines[:first_pipe][0], job_id
    # Behavioral contrast: the same pipeline without pipefail lets tee's zero
    # exit mask the red analyzer.
    run = _analyzer_step("main-receipt")["run"]
    env = {"SOURCE_SHA": "d" * 40, "GITHUB_WORKSPACE": str(tmp_path)}
    with_pipefail = _simulate_step(run, env=env, cwd=tmp_path, stubs={"python3": 9})
    without = _simulate_step(
        run.replace("set -euo pipefail\n", ""), env=env, cwd=tmp_path, stubs={"python3": 9}
    )
    assert with_pipefail.returncode == 9 and without.returncode == 0


def test_contract_drift_outputs_are_required_before_summary(tmp_path):
    terminal = _terminal_step()["run"]
    lines = terminal.splitlines()
    outcome_line = next(index for index, line in enumerate(lines) if "steps.admission.outcome" in line)  # fmt: skip
    output_line = next(index for index, line in enumerate(lines) if "test -s contract-drift-pr-delta.json" in line)  # fmt: skip
    summary_line = next(index for index, line in enumerate(lines) if "accepted_authority" in line)
    assert outcome_line < output_line < summary_line
    empty = _run_terminal(tmp_path / "empty", outcome="success", payload="")
    assert empty.returncode != 0
    present = _run_terminal(tmp_path / "present", outcome="success", payload='{"admitted": true}\n')
    assert present.returncode == 0


def test_skipped_or_cancelled_cdg_authority_job_cannot_report_success(tmp_path):
    # The terminal aggregator turns a skipped admission into a hard failure
    # instead of an implicit green.
    for index, conclusion in enumerate(("skipped", "cancelled")):
        result = _run_terminal(
            tmp_path / str(index), outcome=conclusion, payload='{"admitted": true}\n'
        )
        assert result.returncode != 0


# --- VAL-CDG-009: separate immutable signals (workflow side) ----------------


def test_pull_request_invokes_only_pr_mode():
    assert JOBS["pr-delta"]["if"] == "github.event_name == 'pull_request' && !github.event.pull_request.draft"  # fmt: skip
    admission = _analyzer_step("pr-delta")["run"]
    assert "--mode pr" in admission
    assert "--mode receipt" not in str(JOBS["pr-delta"])
    assert "--mode program" not in str(JOBS["pr-delta"])
    # The two main-only jobs are event-disjoint from PR admission.
    for job_id in ("main-receipt", "program-trajectory"):
        assert JOBS[job_id]["if"] == "github.event_name != 'pull_request'"


def test_pull_request_uses_immutable_event_base_and_head_shas():
    env = JOBS["pr-delta"]["env"]
    assert env == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }
    admission = _analyzer_step("pr-delta")["run"]
    assert admission.index("git fetch") < admission.index("--base-ref")
    assert '"$BASE_SHA" "$HEAD_SHA"' in admission
    # The upload key is bound to the same immutable event head SHA.
    assert _upload_step("pr-delta")["with"]["name"].endswith(
        "${{ github.event.pull_request.head.sha }}"
    )


def test_synthetic_merge_sha_is_rejected_for_pr_delta():
    pr_text = str(JOBS["pr-delta"])
    # The synthetic refs/pull/N/merge commit and its aliases never reach the
    # analyzer: no merge_commit_sha, no github.sha, no mutable ref names.
    assert "merge_commit_sha" not in pr_text
    assert "github.sha" not in pr_text
    assert "refs/pull" not in pr_text and "github.ref" not in pr_text
    admission = _analyzer_step("pr-delta")["run"]
    for flag, value in (("--base-ref", '"$BASE_SHA"'), ("--head-ref", '"$HEAD_SHA"')):
        assert f"{flag} {value}" in admission
    assert "GITHUB_SHA" not in admission and "HEAD^" not in admission


def test_push_main_binds_receipt_and_program_to_event_after_sha():
    assert DOC["on"]["push"]["branches"] == ["main"]
    assert "github.event.after" in SOURCE_SHA_EXPR
    for job_id in ("main-receipt", "program-trajectory"):
        job = JOBS[job_id]
        assert job["env"]["SOURCE_SHA"] == SOURCE_SHA_EXPR
        assert job["steps"][0]["with"]["ref"] == SOURCE_SHA_EXPR
        assert '--ref "$SOURCE_SHA"' in _analyzer_step_text(job)


def test_schedule_and_dispatch_resolve_main_once_for_both_main_jobs():
    assert "schedule" in DOC["on"] and "workflow_dispatch" in DOC["on"]
    # For non-push events the shared expression falls back to the single
    # resolved github.sha, so receipt and trajectory bind one identical SHA.
    assert SOURCE_SHA_EXPR.endswith("|| github.sha }}")
    receipt, program = JOBS["main-receipt"], JOBS["program-trajectory"]
    assert receipt["env"]["SOURCE_SHA"] == program["env"]["SOURCE_SHA"]
    assert receipt["steps"][0]["with"]["ref"] == program["steps"][0]["with"]["ref"]
    assert TEXT.count(SOURCE_SHA_EXPR) == 4


def test_exact_three_live_check_names_are_separate():
    assert len(set(JOBS)) == 3 and len(set(_job_names(DOC))) == 3
    artifact_names = {job_id: _upload_step(job_id)["with"]["name"] for job_id in JOBS}
    assert len(set(artifact_names.values())) == 3
    outputs = {job_id: _upload_step(job_id)["with"]["path"] for job_id in JOBS}
    assert len(set(outputs.values())) == 3
    for job_id in JOBS:
        assert artifact_names[job_id].startswith(f"contract-drift-{job_id}-")
        assert outputs[job_id] == f"contract-drift-{job_id}.json"


def test_terminal_aggregator_fails_when_pr_delta_is_skipped(tmp_path):
    terminal = _terminal_step()
    # `always()` forces the aggregator to run and judge a skipped admission
    # rather than letting the job end green with the step silently absent.
    assert terminal["if"] == "always()"
    assert "steps.admission.outcome" in terminal["run"]
    result = _run_terminal(tmp_path, outcome="skipped", payload='{"admitted": true}\n')
    assert result.returncode == 1
    assert 'test "skipped" = "success"' in terminal["run"].replace(
        "${{ steps.admission.outcome }}", "skipped"
    )


def test_main_receipt_job_is_separate_from_intentionally_red_trajectory():
    receipt, program = JOBS["main-receipt"], JOBS["program-trajectory"]
    assert "needs" not in receipt and "needs" not in program
    assert receipt["name"] == "contract-drift-main-receipt"
    assert program["name"] == "contract-drift-program-trajectory"
    assert "--mode receipt" in _analyzer_step_text(receipt)
    assert "--mode program" in _analyzer_step_text(program)
    assert _upload_step("main-receipt")["with"]["name"] != _upload_step("program-trajectory")["with"]["name"]  # fmt: skip


def test_trajectory_failure_does_not_block_main_receipt(tmp_path):
    # No job anywhere in the workflow declares a dependency on the trajectory
    # job, so its intentionally red exit cannot gate the receipt.
    for job in JOBS.values():
        needs = job.get("needs", [])
        assert "program-trajectory" not in ([needs] if isinstance(needs, str) else needs)
    env = {"SOURCE_SHA": "e" * 40, "GITHUB_WORKSPACE": str(tmp_path)}
    trajectory = _simulate_step(
        _analyzer_step_text(JOBS["program-trajectory"]),
        env=env,
        cwd=tmp_path,
        stubs={"python3": 21},  # fmt: skip
    )
    receipt = _simulate_step(
        _analyzer_step_text(JOBS["main-receipt"]), env=env, cwd=tmp_path, stubs={"python3": 0}
    )
    assert trajectory.returncode == 21 and receipt.returncode == 0
    assert (tmp_path / "contract-drift-main-receipt.json").exists()


def test_main_receipt_success_does_not_mask_trajectory_failure(tmp_path):
    program = JOBS["program-trajectory"]
    assert "continue-on-error" not in str(program)
    # The trajectory job has no aggregator step that could rewrite its
    # conclusion: the analyzer is the last conditioned enforcement point and
    # the only later step is the always-on artifact upload.
    *_, analyzer, upload = program["steps"]
    assert "check_contract_drift_ratchet.py" in analyzer["run"]
    assert upload["uses"].startswith("actions/upload-artifact@")
    env = {"SOURCE_SHA": "f" * 40, "GITHUB_WORKSPACE": str(tmp_path)}
    receipt = _simulate_step(
        _analyzer_step_text(JOBS["main-receipt"]), env=env, cwd=tmp_path, stubs={"python3": 0}
    )
    trajectory = _simulate_step(
        _analyzer_step_text(program), env=env, cwd=tmp_path, stubs={"python3": 13}
    )
    assert receipt.returncode == 0 and trajectory.returncode == 13


def test_program_red_cannot_mask_or_replace_pr_delta_result():
    # PR admission and program trajectory run on disjoint events, publish
    # differently named artifacts, and satisfy different required contexts.
    assert JOBS["pr-delta"]["if"].startswith("github.event_name == 'pull_request'")
    assert JOBS["program-trajectory"]["if"] == "github.event_name != 'pull_request'"
    assert "contract-drift-pr-delta.json" not in str(JOBS["program-trajectory"])
    assert "contract-drift-program-trajectory.json" not in str(JOBS["pr-delta"])


@pytest.mark.skipif(
    os.environ.get("ARAGORA_RUN_LIVE_CDG_WORKFLOW_TEST") != "1",
    reason="set ARAGORA_RUN_LIVE_CDG_WORKFLOW_TEST=1 for authenticated read-only GitHub verification",
)
def test_live_contract_drift_workflow_selection_is_stable_twice():
    first = verify_workflow_state(GhApiReader())
    second = verify_workflow_state(GhApiReader())
    assert first["workflow"] == second["workflow"]
    assert first["main_sha"] == second["main_sha"]
    assert first["selection"]["identity"] == second["selection"]["identity"]
    assert first["stable_requery"] is second["stable_requery"] is True
