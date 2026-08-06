"""VAL-CDG-012 OpenAPI workflow contract: route/SDK exit integrity cannot be masked.

Static and behavioral guards for ``.github/workflows/openapi.yml`` — the single
active OpenAPI workflow (live workflow ID 226588000, required check name
``Generate & Validate``) — plus reference implementations of the live
execution-selection rules: unfiltered pagination ordered by
``(run_started_at, run_id, run_attempt)``, attempt-specific jobs/check URLs,
run-level release-bound artifacts, and the post-verification re-query/restart
rule.

Behavioral tests execute the workflow's actual ``run:`` bodies the way GitHub
does for the default shell (``bash -e``, no implicit pipefail) with
shell-function stubs standing in for named commands, mirroring
tests/ci/test_required_aggregator_fail_closed.py. The selection-rule helpers
mirror ``_paginate_runs`` / ``_plan_date_shards`` / ``_reconcile_run_ids`` /
``_attempt_numbers`` / ``_attempt_jobs_endpoint`` / ``_validate_attempt_jobs``
/ ``_validate_run_artifact`` from tests/scripts/test_contract_drift_workflow.py,
specialized to the OpenAPI workflow. The live GitHub census and execution
selection are exercised by the VAL-CDG-012 measurement itself; these tests pin
the rules and the repo-tree census so a drifting workflow fails here first.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
WORKFLOW_PATH = WORKFLOWS_DIR / "openapi.yml"
TEXT = WORKFLOW_PATH.read_text(encoding="utf-8")
# BaseLoader keeps `on:` a string key and every scalar a string.
DOC = yaml.load(TEXT, Loader=yaml.BaseLoader)
JOBS = DOC["jobs"]

# Live census facts (verified against the GitHub API by the VAL-CDG-012
# measurement; pinned here so the repo-side topology cannot drift silently).
LIVE_CHECK_NAME = "Generate & Validate"
OPENAPI_WORKFLOW_ID = 226588000
RUNS_ENDPOINT_TEMPLATE = (
    f"repos/OWNER/REPO/actions/workflows/{OPENAPI_WORKFLOW_ID}/runs?per_page=100&page={{page}}"
)

# Steps in generate-run whose exit codes carry route-validation,
# SDK-verification, parity, or spec-integrity authority. None of them may mask
# a failure with continue-on-error, `|| true`, or an unconditional-success
# wrapper.
AUTHORITATIVE_STEPS = (
    "Generate OpenAPI spec",
    "Normalize OpenAPI spec metadata",
    "Enforce canonical OpenAPI version",
    "Validate spec structure",
    "Validate YAML output is parseable",
    "AST-only generation smoke test",
    "Generate TypeScript SDK types",
    "Generate Python SDK types",
    "Install OpenAPI validator",
    "Validate OpenAPI spec",
    "Audit documentation coverage",
    "Validate handler route coverage",
    "Check for missing operationIds (warning only)",
    "Verify SDK contracts against spec",
    "Check namespace SDK parity",
    "Run contract matrix tests",
)

# The complete continue-on-error census of the workflow. Every entry is a
# justified non-authoritative seam (see the workflow comments and the rearm PR
# body); anything new appearing here fails this contract until re-adjudicated.
JUSTIFIED_CONTINUE_ON_ERROR = {
    # Runner-resilience seam: the immediately following step re-verifies the
    # toolchain fail-closed under `set -euo pipefail`.
    ("generate-run", "Set up Python 3.11"),
    # PR-lane frontend generated types; the strict twin enforces this
    # generation on every push/schedule/dispatch execution.
    ("generate-run", "Generate Live API types (best-effort on PR)"),
    # Notification only; cannot affect any verdict.
    ("generate-run", "Comment on PR (if spec changed)"),
    # Manual-dispatch publish path, not a validation authority.
    ("sync", "Commit updated spec"),
}

# `|| true` is tolerated only inside non-authoritative recovery/publish steps
# (checkout self-repair, toolchain recovery attempts that are re-verified
# fail-closed, npm cache ownership repair, and the manual-dispatch sync
# publish path). No route validation, SDK verification, or parity step may
# ever appear here.
RECOVERY_STEPS_WITH_OR_TRUE = {
    ("scope", "Verify checkout integrity"),
    ("generate-run", "Verify checkout integrity"),
    ("generate-run", "Ensure python toolchain is available"),
    ("generate-run", "Fix npm cache ownership"),
    ("sync", "Verify checkout integrity"),
    ("sync", "Check for changes"),
    ("sync", "Commit updated spec"),
}

_PIPE = re.compile(r"(?<!\|)\|(?!\|)")


def _steps(job_id: str) -> list[dict]:
    return JOBS[job_id]["steps"]


def _step(job_id: str, name: str) -> dict:
    for step in _steps(job_id):
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} not found in job {job_id!r}")


def _step_index(job_id: str, name: str) -> int:
    for index, step in enumerate(_steps(job_id)):
        if step.get("name") == name:
            return index
    raise AssertionError(f"step {name!r} not found in job {job_id!r}")


def _shell_lines(run: str) -> list[str]:
    """Shell-significant lines of a run block (comment-only lines dropped)."""
    return [line for line in run.splitlines() if not line.strip().startswith("#")]


def _pipeline_lines(run: str) -> list[str]:
    return [line for line in _shell_lines(run) if _PIPE.search(line)]


def _pipefail_armed_before_first_pipe(run: str) -> bool:
    armed = False
    for line in _shell_lines(run):
        if "set -o pipefail" in line or "set -euo pipefail" in line:
            armed = True
        if _PIPE.search(line) and not armed:
            return False
    return True


def _simulate_step(
    run_block: str,
    *,
    env: dict[str, str],
    cwd: Path,
    stub_prelude: str = "",
) -> subprocess.CompletedProcess[str]:
    """Execute a workflow run block the way GitHub's default shell does
    (bash -e, no implicit pipefail), with optional shell-function stubs
    standing in for named commands."""
    return subprocess.run(
        ["bash", "-e", "-c", stub_prelude + run_block],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
    )


def _relocated(run: str, tmp_path: Path) -> str:
    """Relocate the workflow's /tmp scratch paths into the test's tmp dir so
    behavioral simulations stay hermetic. Command structure is unchanged."""
    return run.replace("/tmp/", f"{tmp_path}/")


# ---------------------------------------------------------------------------
# Reference live-selection rules (mirrors of test_contract_drift_workflow.py
# helpers, specialized to the OpenAPI workflow).
# ---------------------------------------------------------------------------


def _paginate_openapi_runs(fetch_page) -> tuple[list[dict], list[str]]:
    """Unfiltered per_page=100 pagination of the OpenAPI workflow's runs,
    terminating on a short page, rejecting duplicate run IDs. The endpoint
    carries no status/conclusion/event/branch filter."""
    records: list[dict] = []
    endpoints: list[str] = []
    page = 1
    while True:
        endpoint = RUNS_ENDPOINT_TEMPLATE.format(page=page)
        for forbidden in ("status=", "conclusion=", "event=", "branch="):
            if forbidden in endpoint:
                raise ValueError("workflow-run selection must be unfiltered")
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


def _attempt_numbers(run: dict) -> list[int]:
    attempt = run.get("run_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("run_attempt is malformed")
    return list(range(1, attempt + 1))


def _selection_key(run: dict) -> tuple[str, int, int]:
    started = run.get("run_started_at")
    if not isinstance(started, str) or not started:
        raise ValueError("run_started_at is malformed")
    run_id = run.get("id")
    if isinstance(run_id, bool) or not isinstance(run_id, int):
        raise ValueError("run id is malformed")
    return (started, run_id, _attempt_numbers(run)[-1])


def _select_newest_execution(records: list[dict]) -> dict:
    """Newest execution by (run_started_at, run_id, run_attempt) over the
    complete unfiltered record set."""
    if not records:
        raise ValueError("no workflow executions to select from")
    return max(records, key=_selection_key)


def _reconcile_total_count(records: list[dict], *, total_count: int | None) -> None:
    if total_count is not None and total_count != len(records):
        raise ValueError("workflow runs do not reconcile to the reported total_count")


def _plan_date_shards(
    daily_counts: dict[str, int], *, cap: int = 1000
) -> list[tuple[str, str, int]]:
    """Disjoint created-date ranges, each strictly below the 1000-result API
    window (only for filtered-history queries; primary selection stays
    unfiltered)."""
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


def _attempt_jobs_endpoint(run_id: int, attempt: int) -> str:
    if attempt < 1:
        raise ValueError("attempt numbers start at 1")
    return f"repos/OWNER/REPO/actions/runs/{run_id}/attempts/{attempt}/jobs"


def _run_artifacts_endpoint(run_id: int) -> str:
    """Artifacts are run-level: the endpoint carries no attempt segment."""
    endpoint = f"repos/OWNER/REPO/actions/runs/{run_id}/artifacts"
    if "/attempts/" in endpoint:
        raise ValueError("artifact queries must be run-level, not attempt-level")
    return endpoint


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


def _selection_is_stable(before: dict, after: dict) -> bool:
    """Post-verification re-query rule: if main, execution identity, attempt,
    or conclusion moved, the whole selection restarts."""
    for key in ("main_sha", "run_id", "run_attempt", "conclusion"):
        if key not in before or key not in after:
            raise ValueError(f"selection snapshot is missing {key!r}")
        if before[key] != after[key]:
            return False
    return True


# ---------------------------------------------------------------------------
# census and trigger topology
# ---------------------------------------------------------------------------


def test_exactly_one_active_openapi_workflow():
    workflow_files = sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))
    assert workflow_files, "no workflow files found"
    named_openapi = [
        path
        for path in workflow_files
        if yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader).get("name")
        == "OpenAPI Spec"
    ]
    assert named_openapi == [WORKFLOW_PATH], (
        "exactly one workflow may declare the OpenAPI Spec pipeline; "
        f"found {[p.name for p in named_openapi]}"
    )
    assert DOC["name"] == "OpenAPI Spec"
    # The pinned live workflow ID is the selection-endpoint constant; the
    # VAL-CDG-012 measurement reconciles it against the GitHub API census
    # (102 workflows, exactly one active OpenAPI workflow).
    assert OPENAPI_WORKFLOW_ID == 226588000
    assert str(OPENAPI_WORKFLOW_ID) in RUNS_ENDPOINT_TEMPLATE


def test_no_duplicate_live_openapi_check_name():
    emitters: list[tuple[str, str]] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml")):
        doc = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        for job_id, job in (doc.get("jobs") or {}).items():
            if isinstance(job, dict) and job.get("name") == LIVE_CHECK_NAME:
                emitters.append((path.name, job_id))
    assert emitters == [("openapi.yml", "generate")], (
        f"the live check name {LIVE_CHECK_NAME!r} must be emitted by exactly one job; "
        f"found {emitters}"
    )


def test_openapi_triggers_pr_main_schedule_dispatch():
    on = DOC["on"]
    assert set(on) == {"pull_request", "push", "schedule", "workflow_dispatch"}
    assert on["pull_request"]["types"] == ["opened", "synchronize", "reopened", "ready_for_review"]
    assert on["pull_request"]["branches"] == ["main"]
    assert on["push"]["branches"] == ["main"]
    assert on["schedule"] == [{"cron": "10 1 * * *"}]
    assert "workflow_dispatch" in on


# ---------------------------------------------------------------------------
# method-aware route and SDK authorities
# ---------------------------------------------------------------------------


def test_openapi_invokes_method_aware_route_and_sdk_authorities():
    route = _step("generate-run", "Validate handler route coverage")["run"]
    # Method-aware plane bound to the resolved exact execution SHA.
    assert 'EXEC_SHA="$(git rev-parse HEAD)"' in route
    assert "scripts/validate_openapi_routes.py" in route
    assert '--ref "${EXEC_SHA}"' in route
    # The workflow verifies the plane echoed the same exact-SHA binding back.
    assert 'plane.get("ref")' in route
    # The baseline-gated coverage check stays the fail-closed drift ratchet.
    assert "--fail-on-missing" in route
    assert "--baseline scripts/baselines/validate_openapi_routes.json" in route
    # The exact-ref refusal and the method-aware plane implementation exist in
    # the invoked authority (superseded heuristic mode is not the binding).
    validator_src = (ROOT / "scripts/validate_openapi_routes.py").read_text(encoding="utf-8")
    assert "_require_exact_ref" in validator_src
    assert "validate_method_aware_plane" in validator_src
    # SDK authorities: strict, baseline-pinned, method-aware by construction
    # (the extraction plane is method-bearing (method, path) operations; see
    # the named-group regexes in the authority source).
    verify = _step("generate-run", "Verify SDK contracts against spec")["run"]
    assert "scripts/verify_sdk_contracts.py" in verify
    assert "--strict" in verify
    assert "--baseline scripts/baselines/verify_sdk_contracts.json" in verify
    parity = _step("generate-run", "Check namespace SDK parity")["run"]
    assert "scripts/check_sdk_namespace_parity.py" in parity
    assert "--strict" in parity
    assert "--baseline scripts/baselines/check_sdk_namespace_parity.json" in parity
    sdk_src = (ROOT / "scripts/verify_sdk_contracts.py").read_text(encoding="utf-8")
    assert "(?P<method>" in sdk_src, "SDK contract authority lost its method-bearing extraction"


# ---------------------------------------------------------------------------
# live execution-selection rules (reference implementations)
# ---------------------------------------------------------------------------


def test_openapi_selects_newest_unfiltered_execution_by_started_run_attempt():
    pages = {
        1: [
            {"id": 100 + i, "run_started_at": f"2026-08-01T00:{i:02d}:00Z", "run_attempt": 1}
            for i in range(100)
        ],
        2: [
            {"id": 300, "run_started_at": "2026-08-02T00:00:00Z", "run_attempt": 2},
            {"id": 299, "run_started_at": "2026-08-02T00:00:00Z", "run_attempt": 1},
            {"id": 200, "run_started_at": "2026-07-31T00:00:00Z", "run_attempt": 5},
        ],
    }

    def fetch_page(endpoint: str) -> list[dict]:
        page = int(endpoint.rsplit("page=", 1)[1])
        return pages.get(page, [])

    records, endpoints = _paginate_openapi_runs(fetch_page)
    assert len(records) == 103
    assert endpoints[0] == RUNS_ENDPOINT_TEMPLATE.format(page=1)
    assert all("status=" not in e and "conclusion=" not in e for e in endpoints)
    _reconcile_total_count(records, total_count=103)
    newest = _select_newest_execution(records)
    # Run 299 shares run_started_at with run 300: the (run_id, run_attempt)
    # tie-breakers select run 300 attempt 2. A later attempt of an older run
    # (id 200, attempt 5) never outranks a newer execution.
    assert (newest["id"], newest["run_attempt"]) == (300, 2)
    with pytest.raises(ValueError, match="reconcile"):
        _reconcile_total_count(records, total_count=104)
    with pytest.raises(ValueError, match="duplicate"):
        _paginate_openapi_runs(
            lambda e: [{"id": 1, "run_started_at": "2026-08-01T00:00:00Z", "run_attempt": 1}] * 2
        )
    # Filtered-history fallback stays sharded strictly below the API window
    # with disjoint date ranges reconciling to the reported totals.
    shards = _plan_date_shards({"2026-08-01": 400, "2026-08-02": 400, "2026-08-03": 400})
    assert all(count < 1000 for _, _, count in shards)
    assert _reconcile_run_ids(
        [{"total_count": 1, "workflow_runs": [{"id": 7}]}], reported_total=1
    ) == [7]
    with pytest.raises(ValueError, match="defeats"):
        _plan_date_shards({"2026-08-01": 1000})


def test_openapi_queries_attempt_specific_jobs_checks_and_run_level_artifacts():
    run = {"id": 300, "run_started_at": "2026-08-02T00:00:00Z", "run_attempt": 3}
    assert _attempt_numbers(run) == [1, 2, 3]
    for attempt in _attempt_numbers(run):
        endpoint = _attempt_jobs_endpoint(300, attempt)
        jobs = [
            {
                "run_attempt": attempt,
                "check_url": f"repos/OWNER/REPO/check-runs/9/attempts/{attempt}",
            }
        ]
        _validate_attempt_jobs(endpoint, jobs, attempt=attempt)
    with pytest.raises(ValueError, match="attempt-specific"):
        _validate_attempt_jobs("repos/OWNER/REPO/actions/runs/300/jobs", [], attempt=2)
    with pytest.raises(ValueError, match="not attempt-specific"):
        _validate_attempt_jobs(
            _attempt_jobs_endpoint(300, 2),
            [{"run_attempt": 1, "check_url": "repos/OWNER/REPO/check-runs/9/attempts/2"}],
            attempt=2,
        )
    with pytest.raises(ValueError, match="malformed"):
        _attempt_numbers({"run_attempt": 0})
    artifacts_endpoint = _run_artifacts_endpoint(300)
    assert artifacts_endpoint.endswith("/actions/runs/300/artifacts")
    head = "a" * 40
    good = {"name": f"openapi-spec-{head}", "size_in_bytes": 10, "payload_sha256": "d" * 64}
    assert _validate_run_artifact(good, head_sha=head, release_digests={"d" * 64})
    with pytest.raises(ValueError, match="SHA-bound"):
        _validate_run_artifact(
            {"name": "openapi-spec", "size_in_bytes": 10, "payload_sha256": "d" * 64},
            head_sha=head,
            release_digests={"d" * 64},
        )
    with pytest.raises(ValueError, match="nonempty"):
        _validate_run_artifact(
            {"name": f"openapi-spec-{head}", "size_in_bytes": 0, "payload_sha256": "d" * 64},
            head_sha=head,
            release_digests={"d" * 64},
        )
    with pytest.raises(ValueError, match="immutable release"):
        _validate_run_artifact(
            {"name": f"openapi-spec-{head}", "size_in_bytes": 10, "payload_sha256": "e" * 64},
            head_sha=head,
            release_digests={"d" * 64},
        )


def test_openapi_restarts_if_main_execution_attempt_or_conclusion_moves():
    before = {"main_sha": "a" * 40, "run_id": 300, "run_attempt": 2, "conclusion": "success"}
    assert _selection_is_stable(before, dict(before))
    for key, moved in (
        ("main_sha", "b" * 40),
        ("run_id", 301),
        ("run_attempt", 3),
        ("conclusion", "failure"),
    ):
        after = dict(before)
        after[key] = moved
        assert not _selection_is_stable(before, after), f"movement in {key} must restart selection"
    with pytest.raises(ValueError, match="missing"):
        _selection_is_stable(before, {"main_sha": "a" * 40})


# ---------------------------------------------------------------------------
# authoritative steps propagate failure; pipefail before every pipeline
# ---------------------------------------------------------------------------


def test_openapi_authoritative_route_and_sdk_steps_propagate_nonzero_exit(tmp_path):
    for name in AUTHORITATIVE_STEPS:
        step = _step("generate-run", name)
        assert "continue-on-error" not in step, f"{name!r} may not continue-on-error"
        assert "always()" not in str(step.get("if", "")), f"{name!r} may not run unconditionally"
        assert "|| true" not in step["run"], f"{name!r} may not mask its exit with || true"
    # continue-on-error census: every occurrence is a justified
    # non-authoritative seam, and nothing else.
    census = {
        (job_id, step.get("name"))
        for job_id, job in JOBS.items()
        for step in job.get("steps", [])
        if "continue-on-error" in step
    }
    assert census == JUSTIFIED_CONTINUE_ON_ERROR
    # `|| true` census: non-authoritative recovery/publish steps only.
    or_true = {
        (job_id, step.get("name"))
        for job_id, job in JOBS.items()
        for step in job.get("steps", [])
        if "|| true" in step.get("run", "")
    }
    assert or_true <= RECOVERY_STEPS_WITH_OR_TRUE
    # The setup-python seam is closed by a fail-closed toolchain verification
    # immediately after it.
    steps = _steps("generate-run")
    setup_index = _step_index("generate-run", "Set up Python 3.11")
    ensure = steps[setup_index + 1]
    assert ensure["name"] == "Ensure python toolchain is available"
    assert "set -euo pipefail" in ensure["run"] and "exit 1" in ensure["run"]
    assert "continue-on-error" not in ensure
    # The PR-lane best-effort Live types step has a strict non-PR twin.
    strict_live = _step("generate-run", "Generate Live API types (strict)")
    assert strict_live["if"] == "github.event_name != 'pull_request'"
    assert "continue-on-error" not in strict_live
    best_effort = _step("generate-run", "Generate Live API types (best-effort on PR)")
    assert best_effort["if"] == "github.event_name == 'pull_request'"
    # Behavioral: a failing route authority fails the block even though a
    # successful consumer (tee) sits after it in the pipeline.
    route = _relocated(_step("generate-run", "Validate handler route coverage")["run"], tmp_path)
    fail_stub = (
        "git() { echo 0123456789abcdef0123456789abcdef01234567; }\n"
        "tee() { cat >/dev/null; return 0; }\n"
        'python() { if [ "${1:-}" = "scripts/validate_openapi_routes.py" ]; then return 1; fi; '
        "cat >/dev/null 2>/dev/null || true; return 0; }\n"
    )
    result = _simulate_step(route, env={}, cwd=tmp_path, stub_prelude=fail_stub)
    assert result.returncode != 0, "route authority failure was masked by the tee consumer"
    ok_stub = (
        "git() { echo 0123456789abcdef0123456789abcdef01234567; }\n"
        "tee() { cat >/dev/null; return 0; }\n"
        "python() { cat >/dev/null 2>/dev/null || true; return 0; }\n"
    )
    result = _simulate_step(route, env={}, cwd=tmp_path, stub_prelude=ok_stub)
    assert result.returncode == 0, result.stderr
    # Behavioral: the contract-matrix pytest exit code decides its step.
    matrix = _relocated(_step("generate-run", "Run contract matrix tests")["run"], tmp_path)
    for pytest_rc, expect_failure in ((1, True), (0, False)):
        stub = f"python() {{ return 0; }}\npytest() {{ return {pytest_rc}; }}\n"
        result = _simulate_step(matrix, env={}, cwd=tmp_path, stub_prelude=stub)
        assert (result.returncode != 0) is expect_failure, (
            f"contract-matrix rc={pytest_rc} must propagate; got {result.returncode}"
        )
    # Behavioral: a failing SDK verification authority fails its block.
    verify = _relocated(_step("generate-run", "Verify SDK contracts against spec")["run"], tmp_path)
    stub = "python() { return 1; }\n"
    result = _simulate_step(verify, env={}, cwd=tmp_path, stub_prelude=stub)
    assert result.returncode != 0, "SDK contract verification failure was masked"


def test_every_openapi_authoritative_pipeline_enables_pipefail_before_first_pipe():
    piped = []
    for name in AUTHORITATIVE_STEPS:
        run = _step("generate-run", name)["run"]
        if not _pipeline_lines(run):
            continue
        piped.append(name)
        assert _pipefail_armed_before_first_pipe(run), (
            f"{name!r} contains a pipeline before pipefail is armed"
        )
    # The census of authoritative piped steps must include the known two; a
    # new pipeline elsewhere is caught by the loop above.
    assert "Validate handler route coverage" in piped
    assert "Check for missing operationIds (warning only)" in piped
    # The gate aggregator arms pipefail unconditionally before any logic.
    gate_run = _steps("generate")[0]["run"]
    assert "set -euo pipefail" in gate_run


# ---------------------------------------------------------------------------
# summaries require parseable current-run output
# ---------------------------------------------------------------------------


def test_openapi_outputs_are_required_before_summary(tmp_path):
    route_index = _step_index("generate-run", "Validate handler route coverage")
    summary_index = _step_index("generate-run", "Generate contract drift summary")
    backlog_index = _step_index("generate-run", "Generate contract drift backlog seeds")
    assert route_index < summary_index < backlog_index
    summary = _step("generate-run", "Generate contract drift summary")["run"]
    backlog = _step("generate-run", "Generate contract drift backlog seeds")["run"]
    # Each summary step demands parseable current-run output BEFORE running
    # its generator (the generator is what presents the verdict).
    assert summary.index("json.load") < summary.index("scripts/contract_drift_report.py")
    assert backlog.index("json.load") < backlog.index("scripts/generate_contract_drift_backlog.py")
    # Behavioral: missing or unparseable current-run output fails the step
    # instead of presenting a verdict from stale inputs. The generator
    # scripts are stubbed; the json.load preconditions run under real python.
    stub = 'python() { case "${1:-}" in scripts/*) return 0;; *) command python3 "$@";; esac; }\n'
    summary_run = _relocated(summary, tmp_path)
    backlog_run = _relocated(backlog, tmp_path)
    result = _simulate_step(summary_run, env={}, cwd=tmp_path, stub_prelude=stub)
    assert result.returncode != 0, "summary presented a verdict without current-run output"
    (tmp_path / "route-coverage.json").write_text("{}", encoding="utf-8")
    api_dir = tmp_path / "docs" / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "openapi_generated.json").write_text("{}", encoding="utf-8")
    result = _simulate_step(summary_run, env={}, cwd=tmp_path, stub_prelude=stub)
    assert result.returncode == 0, result.stderr
    result = _simulate_step(backlog_run, env={}, cwd=tmp_path, stub_prelude=stub)
    assert result.returncode != 0, "backlog seeded a verdict without the current-run summary"
    (tmp_path / "contract-drift-summary.json").write_text("{}", encoding="utf-8")
    result = _simulate_step(backlog_run, env={}, cwd=tmp_path, stub_prelude=stub)
    assert result.returncode == 0, result.stderr
    # Truncated current-run output is unparseable and must also fail.
    (tmp_path / "route-coverage.json").write_text('{"coverage', encoding="utf-8")
    result = _simulate_step(summary_run, env={}, cwd=tmp_path, stub_prelude=stub)
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# the gate and the summary/artifact steps
# ---------------------------------------------------------------------------


def test_openapi_gate_has_no_continue_on_error_or_unconditional_success(tmp_path):
    gate = JOBS["generate"]
    assert gate["name"] == LIVE_CHECK_NAME
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == {"scope", "generate-run"}
    assert "continue-on-error" not in str(gate)
    (step,) = gate["steps"]
    run = step["run"]
    assert "|| true" not in run
    env_map = step["env"]
    assert env_map["CLASSIFIER_RESULT"] == "${{ needs.scope.result }}"
    assert env_map["WORKER_RESULT"] == "${{ needs.generate-run.result }}"
    assert env_map["IN_SCOPE"] == "${{ needs.scope.outputs.run_openapi }}"
    cases = [
        # (classifier, worker, in_scope, expect_failure)
        ("success", "success", "true", False),
        ("success", "failure", "true", True),
        ("success", "cancelled", "true", True),
        ("success", "timed_out", "true", True),
        ("success", "skipped", "true", True),  # required work never ran
        ("success", "skipped", "false", False),  # legitimate out-of-scope skip
        ("success", "success", "false", True),  # out of scope yet ran: distrust
        ("success", "failure", "false", True),
        ("failure", "success", "true", True),  # classifier verdict untrusted
        ("failure", "skipped", "false", True),
        ("cancelled", "skipped", "false", True),
        ("skipped", "skipped", "false", True),
    ]
    for classifier, worker, in_scope, expect_failure in cases:
        result = _simulate_step(
            run,
            env={"CLASSIFIER_RESULT": classifier, "WORKER_RESULT": worker, "IN_SCOPE": in_scope},
            cwd=tmp_path,
        )
        assert (result.returncode != 0) is expect_failure, (
            f"gate(classifier={classifier}, worker={worker}, in_scope={in_scope}) "
            f"returned {result.returncode}"
        )


def test_openapi_summary_and_artifact_steps_cannot_rewrite_gate_conclusion():
    # The gate decides only from its dependencies' results and outputs.
    (gate_step,) = JOBS["generate"]["steps"]
    expressions = re.findall(r"\$\{\{\s*([^}]+?)\s*\}\}", str(gate_step))
    allowed = {
        "needs.scope.result",
        "needs.generate-run.result",
        "needs.scope.outputs.run_openapi",
    }
    assert set(expressions) <= allowed, f"gate reads unsupported expressions: {expressions}"
    # Summary/backlog steps may run after failure but cannot mask their own
    # failure, and never touch job outputs or conclusions.
    for name in ("Generate contract drift summary", "Generate contract drift backlog seeds"):
        step = _step("generate-run", name)
        assert step.get("if") == "always()"
        assert "continue-on-error" not in step
        assert "GITHUB_OUTPUT" not in step["run"] and "gh api" not in step["run"]
    # Artifact uploads carry only name/path/retention and cannot reinterpret
    # any outcome.
    uploads = [
        step
        for step in _steps("generate-run")
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 2
    for step in uploads:
        assert step["uses"] == "actions/upload-artifact@v4"
        assert set(step["with"]) <= {"name", "path", "retention-days"}
        assert "continue-on-error" not in step
        assert step.get("if") in (None, "always()")
    # No step anywhere in the workflow writes check runs, check conclusions,
    # or commit statuses.
    assert "checks.create" not in TEXT
    assert "check-runs" not in TEXT
    assert "statuses/" not in TEXT
    # The PR comment step is notification-only.
    comment = _step("generate-run", "Comment on PR (if spec changed)")
    script = comment["with"]["script"]
    assert "createComment" in script
    assert "conclusion" not in script
