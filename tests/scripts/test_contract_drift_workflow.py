import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

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
    return JOBS["pr-delta"]["steps"][-1]


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
    return subprocess.run(
        ["bash", "-e", "-c", prelude + run_block],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        cwd=str(cwd),
    )


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


def _verify_protection_cutover(
    before: dict,
    after: dict,
    *,
    added_context: str = "contract-drift-pr-delta",
    expected_app_id: int = EXPECTED_PR_DELTA_APP_ID,
) -> list[tuple[str, int]]:
    """Reference cutover rule: after == before plus exactly the pr-delta
    (context, app_id) tuple, with strict preserved and nothing mutated."""
    if before["strict"] != after["strict"]:
        raise ValueError("cutover changed required_status_checks.strict")
    before_tuples = [tuple(item) for item in before["checks"]]
    after_tuples = [tuple(item) for item in after["checks"]]
    if len(before_tuples) != len(set(before_tuples)) or len(after_tuples) != len(set(after_tuples)):
        raise ValueError("required check tuples are duplicated")
    removed = set(before_tuples) - set(after_tuples)
    if removed:
        raise ValueError("cutover removed or mutated a required (context, app_id) tuple")
    added = set(after_tuples) - set(before_tuples)
    if added != {(added_context, expected_app_id)}:
        raise ValueError(
            "cutover must add exactly the pr-delta context with the expected app identity"
        )
    return sorted(set(after_tuples))


def _paginated_protection_checks(pages: list[list[tuple[str, int]]]) -> list[tuple[str, int]]:
    checks: list[tuple[str, int]] = []
    for page in pages:
        if len(page) > 100:
            raise ValueError("paginated protection page exceeds per_page=100")
        checks.extend((str(context), int(app_id)) for context, app_id in page)
    if len(checks) != len(set(checks)):
        raise ValueError("paginated protection capture returned duplicate tuples")
    return checks


def _required_check_satisfied(
    check_runs: list[dict], *, context: str, app_id: int = EXPECTED_PR_DELTA_APP_ID
) -> bool:
    """A required context is satisfied only by its own successful check run;
    no other check's state can substitute for it."""
    for run in check_runs:
        if run.get("name") == context and run.get("app_id") == app_id:
            return run.get("conclusion") == "success"
    return False


def _job_reports_success(job: dict) -> bool:
    if job.get("conclusion") in {"skipped", "cancelled"}:
        return False
    return job.get("conclusion") == "success"


def _paginate_runs(fetch_page) -> tuple[list[dict], list[str]]:
    """Reference workflow-run pagination: unfiltered per_page=100 pages,
    terminating on a short page, rejecting duplicate run IDs."""
    records: list[dict] = []
    endpoints: list[str] = []
    page = 1
    while True:
        endpoint = f"repos/OWNER/REPO/actions/runs?per_page=100&page={page}"
        endpoints.append(endpoint)
        payload = fetch_page(endpoint)
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("paginated workflow-run payload is malformed")
        records.extend(payload)
        if len(payload) < 100:
            break
        page += 1
        if page > 10_000:
            raise ValueError("workflow-run pagination did not terminate")
    ids = [run.get("id") for run in records]
    if len(ids) != len(set(ids)):
        raise ValueError("paginated workflow runs returned duplicate record IDs")
    return records, endpoints


def _plan_date_shards(
    daily_counts: dict[str, int], *, cap: int = 1000
) -> list[tuple[str, str, int]]:
    """Reference shard planner: disjoint created-date ranges, each strictly
    below the 1000-result API window."""
    shards: list[tuple[str, str, int]] = []
    current: list[str] = []
    total = 0
    for day in sorted(daily_counts):
        count = daily_counts[day]
        if count >= cap:
            raise ValueError(f"single-day run volume {count} defeats the {cap}-result window")
        if current and total + count >= cap:
            shards.append((current[0], current[-1], total))
            current, total = [], 0
        current.append(day)
        total += count
    if current:
        shards.append((current[0], current[-1], total))
    for (_, end_a, _), (start_b, _, _) in zip(shards, shards[1:]):
        if not end_a < start_b:
            raise ValueError("date shards overlap")
    if any(count >= cap for _, _, count in shards):
        raise ValueError(f"date shard reaches the {cap}-result cap")
    return shards


def _reconcile_run_ids(shards: list[dict], *, reported_total: int) -> list[int]:
    ids = [run["id"] for shard in shards for run in shard["workflow_runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("sharded run capture duplicated run IDs")
    for shard in shards:
        if shard["total_count"] != len(shard["workflow_runs"]):
            raise ValueError("shard total_count does not reconcile with captured runs")
    if reported_total != len(ids):
        raise ValueError("run IDs do not reconcile to the reported total_count")
    return sorted(ids)


def _attempt_numbers(run: dict) -> list[int]:
    attempt = run.get("run_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("run_attempt is malformed")
    return list(range(1, attempt + 1))


def _attempt_jobs_endpoint(run_id: int, attempt: int) -> str:
    if attempt < 1:
        raise ValueError("attempt numbers start at 1")
    return f"repos/OWNER/REPO/actions/runs/{run_id}/attempts/{attempt}/jobs"


def _validate_attempt_jobs(endpoint: str, jobs: list[dict], *, attempt: int) -> None:
    if f"/attempts/{attempt}/jobs" not in endpoint:
        raise ValueError("jobs endpoint is not attempt-specific")
    for job in jobs:
        if job.get("run_attempt") != attempt:
            raise ValueError("job record is not attempt-specific")
        if f"/attempts/{attempt}" not in job.get("check_url", ""):
            raise ValueError("check URL is not pinned to the requested attempt")


def _validate_run_artifact(artifact: dict, *, head_sha: str, release_digests: set[str]) -> str:
    name = artifact.get("name")
    if not isinstance(name, str) or not name.endswith(f"-{head_sha}"):
        raise ValueError("run artifact name is not SHA-bound")
    size = artifact.get("size_in_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("run artifact lacks a nonempty payload")
    if artifact.get("payload_sha256") not in release_digests:
        raise ValueError("run artifact payload is not bound to the immutable release")
    return name


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


def _workflow_docs() -> dict[str, dict]:
    docs = {}
    for path in sorted((ROOT / ".github/workflows").iterdir()):
        if path.suffix in {".yml", ".yaml"}:
            loaded = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
            if isinstance(loaded, dict):
                docs[path.name] = loaded
    return docs


def _job_names(doc: dict) -> list[str]:
    return [
        str(job.get("name", job_id))
        for job_id, job in doc.get("jobs", {}).items()
        if isinstance(job, dict)
    ]


def test_exactly_one_active_contract_drift_workflow():
    producers = [
        name
        for name, doc in _workflow_docs().items()
        if set(_job_names(doc)) & set(LIVE_CHECK_NAMES)
    ]
    assert producers == ["contract-drift-governance.yml"]
    doc = _workflow_docs()["contract-drift-governance.yml"]
    assert doc["name"] == "Contract Drift Governance"
    # The producer is active: real triggers, and no workflow-level disable.
    assert doc["on"] and "jobs" in doc


def test_exact_three_live_cdg_check_names():
    names = _job_names(DOC)
    assert sorted(names) == sorted(LIVE_CHECK_NAMES)
    # Job IDs map 1:1 onto the exact live check names.
    assert {job_id: job["name"] for job_id, job in JOBS.items()} == {
        "pr-delta": "contract-drift-pr-delta",
        "main-receipt": "contract-drift-main-receipt",
        "program-trajectory": "contract-drift-program-trajectory",
    }


def test_no_duplicate_live_cdg_check_names():
    names = _job_names(DOC)
    assert len(names) == len(set(names)) == 3
    for live in LIVE_CHECK_NAMES:
        producers = [name for name, doc in _workflow_docs().items() if _job_names(doc).count(live)]
        assert producers == ["contract-drift-governance.yml"], live
        assert _job_names(DOC).count(live) == 1


def test_contract_drift_triggers_pr_main_schedule_dispatch():
    assert set(DOC["on"]) == {"pull_request", "push", "schedule", "workflow_dispatch"}
    assert DOC["on"]["pull_request"]["branches"] == ["main"]
    assert DOC["on"]["push"]["branches"] == ["main"]
    schedule = DOC["on"]["schedule"]
    assert schedule and all("cron" in entry for entry in schedule)
    assert DOC["on"]["workflow_dispatch"] == {}


def test_contract_drift_pr_has_no_governed_path_filter_gap():
    for event in ("pull_request", "push"):
        trigger = DOC["on"][event]
        assert not {"paths", "paths-ignore"} & set(trigger), event
    assert "paths" not in TEXT and "paths-ignore" not in TEXT


def test_contract_drift_pr_uses_event_bound_full_shas():
    env = JOBS["pr-delta"]["env"]
    assert env["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert env["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    admission = _analyzer_step("pr-delta")["run"]
    assert '--base-ref "$BASE_SHA"' in admission and '--head-ref "$HEAD_SHA"' in admission
    assert 'git fetch --no-tags origin "$BASE_SHA" "$HEAD_SHA"' in admission
    # No mutable or synthetic refs stand in for the immutable event SHAs.
    pr_text = str(JOBS["pr-delta"])
    assert "github.ref" not in pr_text and "merge_commit_sha" not in pr_text
    assert "--base-ref HEAD" not in pr_text and "--head-ref HEAD" not in pr_text


def test_contract_drift_non_pr_events_resolve_one_sha_for_receipt_and_program():
    receipt, program = JOBS["main-receipt"], JOBS["program-trajectory"]
    assert receipt["env"]["SOURCE_SHA"] == SOURCE_SHA_EXPR
    assert program["env"]["SOURCE_SHA"] == SOURCE_SHA_EXPR
    for job in (receipt, program):
        checkout = job["steps"][0]
        assert checkout["uses"].startswith("actions/checkout@")
        assert checkout["with"]["ref"] == SOURCE_SHA_EXPR
        assert '--ref "$SOURCE_SHA"' in _analyzer_step_text(job)
        assert job["if"] == "github.event_name != 'pull_request'"


def _analyzer_step_text(job: dict) -> str:
    for step in job["steps"]:
        if "run" in step and "check_contract_drift_ratchet.py" in step["run"]:
            return step["run"]
    raise AssertionError("no analyzer step")


def test_workflow_runs_paginate_unfiltered_and_filter_locally():
    runs = [{"id": index, "name": DOC["name"] if index % 3 else "Other"} for index in range(237)]

    def fetch_page(endpoint: str) -> list[dict]:
        # The listing endpoint carries pagination only: any server-side
        # filter (status/event/created/branch) would hide history.
        query = endpoint.split("?", 1)[1]
        assert sorted(param.split("=")[0] for param in query.split("&")) == ["page", "per_page"]
        page = int(query.split("page=")[-1])
        return runs[(page - 1) * 100 : page * 100]

    records, endpoints = _paginate_runs(fetch_page)
    assert [len(records), len(endpoints)] == [237, 3]
    assert endpoints[0].endswith("per_page=100&page=1")
    governance = [run for run in records if run["name"] == DOC["name"]]
    assert len(governance) == 158 and all(run["id"] % 3 for run in governance)
    with pytest.raises(ValueError, match="duplicate record IDs"):
        _paginate_runs(
            lambda endpoint: [{"id": 1}] * 100 if endpoint.endswith("&page=1") else [{"id": 1}]
        )


def test_filtered_history_requires_disjoint_date_shards_below_1000():
    shards = _plan_date_shards({f"2026-07-{day:02d}": 400 for day in range(1, 8)})
    assert shards == [
        ("2026-07-01", "2026-07-02", 800),
        ("2026-07-03", "2026-07-04", 800),
        ("2026-07-05", "2026-07-06", 800),
        ("2026-07-07", "2026-07-07", 400),
    ]
    assert all(count < 1000 for _, _, count in shards)
    starts = [start for start, _, _ in shards][1:]
    ends = [end for _, end, _ in shards][:-1]
    assert all(end < start for end, start in zip(ends, starts))
    with pytest.raises(ValueError, match="defeats the 1000-result window"):
        _plan_date_shards({"2026-07-01": 1000})


def test_workflow_run_ids_reconcile_to_total_count():
    shards = [
        {"total_count": 2, "workflow_runs": [{"id": 11}, {"id": 12}]},
        {"total_count": 1, "workflow_runs": [{"id": 13}]},
    ]
    assert _reconcile_run_ids(shards, reported_total=3) == [11, 12, 13]
    with pytest.raises(ValueError, match="do not reconcile to the reported total_count"):
        _reconcile_run_ids(shards, reported_total=4)
    with pytest.raises(ValueError, match="duplicated run IDs"):
        _reconcile_run_ids(
            [
                {"total_count": 1, "workflow_runs": [{"id": 11}]},
                {"total_count": 1, "workflow_runs": [{"id": 11}]},
            ],
            reported_total=2,
        )
    with pytest.raises(ValueError, match="shard total_count does not reconcile"):
        _reconcile_run_ids([{"total_count": 5, "workflow_runs": [{"id": 11}]}], reported_total=1)


def test_attempts_are_enumerated_one_through_run_attempt():
    assert _attempt_numbers({"run_attempt": 1}) == [1]
    assert _attempt_numbers({"run_attempt": 4}) == [1, 2, 3, 4]
    for hostile in ({"run_attempt": 0}, {"run_attempt": True}, {"run_attempt": "3"}, {}):
        with pytest.raises(ValueError, match="run_attempt is malformed"):
            _attempt_numbers(hostile)


def test_jobs_and_check_urls_are_attempt_specific():
    endpoint = _attempt_jobs_endpoint(9662, 2)
    assert endpoint == "repos/OWNER/REPO/actions/runs/9662/attempts/2/jobs"
    jobs = [{"run_attempt": 2, "check_url": "repos/OWNER/REPO/actions/runs/9662/attempts/2/jobs/7"}]
    _validate_attempt_jobs(endpoint, jobs, attempt=2)
    with pytest.raises(ValueError, match="not attempt-specific"):
        _validate_attempt_jobs("repos/OWNER/REPO/actions/runs/9662/jobs", jobs, attempt=2)
    with pytest.raises(ValueError, match="job record is not attempt-specific"):
        _validate_attempt_jobs(endpoint, [{"run_attempt": 1, "check_url": endpoint}], attempt=2)
    with pytest.raises(ValueError, match="not pinned to the requested attempt"):
        _validate_attempt_jobs(
            endpoint,
            [{"run_attempt": 2, "check_url": "repos/OWNER/REPO/actions/runs/9662/jobs/7"}],
            attempt=2,
        )
    with pytest.raises(ValueError, match="attempt numbers start at 1"):
        _attempt_jobs_endpoint(9662, 0)


def test_run_level_artifacts_require_payload_and_release_binding():
    head = "a" * 40
    artifact = {
        "name": f"contract-drift-pr-delta-{head}",
        "size_in_bytes": 2048,
        "payload_sha256": "d" * 64,
    }
    assert _validate_run_artifact(artifact, head_sha=head, release_digests={"d" * 64})
    with pytest.raises(ValueError, match="nonempty payload"):
        _validate_run_artifact({**artifact, "size_in_bytes": 0}, head_sha=head, release_digests={"d" * 64})  # fmt: skip
    with pytest.raises(ValueError, match="not bound to the immutable release"):
        _validate_run_artifact(artifact, head_sha=head, release_digests={"e" * 64})
    with pytest.raises(ValueError, match="not SHA-bound"):
        _validate_run_artifact({**artifact, "name": "contract-drift-pr-delta"}, head_sha=head, release_digests={"d" * 64})  # fmt: skip
    # Every live job uploads a run-level, SHA-qualified copy of its output.
    for job_id, sha_expr in (
        ("pr-delta", "${{ github.event.pull_request.head.sha }}"),
        ("main-receipt", "${{ github.sha }}"),
        ("program-trajectory", "${{ github.sha }}"),
    ):
        upload = _upload_step(job_id)
        assert upload["with"]["name"] == f"contract-drift-{job_id}-{sha_expr}"
        assert upload["with"]["path"] == f"contract-drift-{job_id}.json"


def test_pr_delta_cutover_preserves_exact_context_app_id_set_and_strict():
    before = {"strict": True, "checks": [["ci/build", 15368], ["ci/lint", 15368]]}
    after = {
        "strict": True,
        "checks": [["ci/build", 15368], ["ci/lint", 15368], ["contract-drift-pr-delta", 15368]],
    }
    result = _verify_protection_cutover(before, after)
    assert ("contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID) in result
    assert set(result) >= {("ci/build", 15368), ("ci/lint", 15368)}
    with pytest.raises(ValueError, match="changed required_status_checks.strict"):
        _verify_protection_cutover(before, {**after, "strict": False})
    checks = _paginated_protection_checks([[("ci/build", 15368)], [("ci/lint", 15368)]])
    assert checks == [("ci/build", 15368), ("ci/lint", 15368)]
    with pytest.raises(ValueError, match="duplicate tuples"):
        _paginated_protection_checks([[("ci/build", 15368)], [("ci/build", 15368)]])


def test_pr_delta_uses_expected_app_identity():
    runs = [
        {
            "name": "contract-drift-pr-delta",
            "app_id": EXPECTED_PR_DELTA_APP_ID,
            "conclusion": "success",
        }
    ]
    assert _required_check_satisfied(runs, context="contract-drift-pr-delta")
    imposter = [{"name": "contract-drift-pr-delta", "app_id": 99999, "conclusion": "success"}]
    assert not _required_check_satisfied(imposter, context="contract-drift-pr-delta")
    before = {"strict": True, "checks": [["ci/build", 15368]]}
    after = {"strict": True, "checks": [["ci/build", 15368], ["contract-drift-pr-delta", 99999]]}
    with pytest.raises(ValueError, match="expected app identity"):
        _verify_protection_cutover(before, after)


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
    for conclusion in ("skipped", "cancelled"):
        assert not _job_reports_success({"conclusion": conclusion})
        assert not _required_check_satisfied(
            [
                {
                    "name": "contract-drift-pr-delta",
                    "app_id": EXPECTED_PR_DELTA_APP_ID,
                    "conclusion": conclusion,
                }
            ],
            context="contract-drift-pr-delta",
        )
    assert _job_reports_success({"conclusion": "success"})
    assert not _job_reports_success({"conclusion": None})
    # The terminal aggregator turns a skipped admission into a hard failure
    # instead of an implicit green.
    result = _run_terminal(tmp_path, outcome="skipped", payload='{"admitted": true}\n')
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


def test_pr_delta_becomes_branch_protection_required_after_cutover():
    before = {"strict": True, "checks": [["ci/build", 15368]]}
    after = {
        "strict": True,
        "checks": [["ci/build", 15368], ["contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID]],
    }
    result = _verify_protection_cutover(before, after)
    assert ("contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID) in result
    # A cutover that fails to add the pr-delta requirement is rejected.
    with pytest.raises(ValueError, match="must add exactly the pr-delta context"):
        _verify_protection_cutover(before, before)


def test_branch_protection_preserves_exact_context_app_id_set_and_strict():
    before = {"strict": True, "checks": [["ci/build", 15368], ["ci/lint", 15368]]}
    dropped = {
        "strict": True,
        "checks": [["ci/build", 15368], ["contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID]],
    }
    with pytest.raises(ValueError, match="removed or mutated"):
        _verify_protection_cutover(before, dropped)
    swapped_app = {
        "strict": True,
        "checks": [
            ["ci/build", 15368],
            ["ci/lint", 424242],
            ["contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID],
        ],
    }
    with pytest.raises(ValueError, match="removed or mutated"):
        _verify_protection_cutover(before, swapped_app)
    duplicated = {
        "strict": True,
        "checks": [["ci/build", 15368], ["ci/build", 15368]],
    }
    with pytest.raises(ValueError, match="duplicated"):
        _verify_protection_cutover(before, duplicated)


def test_pr_delta_is_added_with_expected_app_identity():
    before = {"strict": False, "checks": [["ci/build", 15368]]}
    for hostile_app in (99999, 0, -1):
        after = {
            "strict": False,
            "checks": [["ci/build", 15368], ["contract-drift-pr-delta", hostile_app]],
        }
        with pytest.raises(ValueError, match="expected app identity"):
            _verify_protection_cutover(before, after)
    good = {
        "strict": False,
        "checks": [["ci/build", 15368], ["contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID]],
    }
    assert ("contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID) in _verify_protection_cutover(
        before, good
    )
    # Adding a different context, even with the right app, is not the cutover.
    wrong_context = {
        "strict": False,
        "checks": [["ci/build", 15368], ["contract-drift-imposter", EXPECTED_PR_DELTA_APP_ID]],
    }
    with pytest.raises(ValueError, match="must add exactly the pr-delta context"):
        _verify_protection_cutover(before, wrong_context)


def test_terminal_aggregator_fails_when_pr_delta_is_skipped(tmp_path):
    terminal = _terminal_step()
    # `always()` forces the aggregator to run and judge a skipped admission
    # rather than letting the job end green with the step silently absent.
    assert terminal["if"] == "always()"
    assert terminal is JOBS["pr-delta"]["steps"][-1]
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
    trajectory_success = [
        {
            "name": "contract-drift-program-trajectory",
            "app_id": EXPECTED_PR_DELTA_APP_ID,
            "conclusion": "success",
        }
    ]
    assert not _required_check_satisfied(trajectory_success, context="contract-drift-pr-delta")
    red_delta = trajectory_success + [
        {
            "name": "contract-drift-pr-delta",
            "app_id": EXPECTED_PR_DELTA_APP_ID,
            "conclusion": "failure",
        }
    ]
    assert not _required_check_satisfied(red_delta, context="contract-drift-pr-delta")
