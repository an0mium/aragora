from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "contract-drift-governance.yml"


def _contract() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    assert len(jobs) == 3
    assert {job["name"] for job in jobs.values()} == {
        f"contract-drift-{name}" for name in ("pr-delta", "main-receipt", "program-trajectory")
    }
    assert set(workflow["on"]) == {"pull_request", "push", "schedule", "workflow_dispatch"}
    assert not ({"paths", "paths-ignore"} & set(workflow["on"]["pull_request"]))
    text = WORKFLOW.read_text()
    assert text.count("github.event.pull_request.") >= 2
    assert '--base-ref "$BASE_SHA"' in text and '--head-ref "$HEAD_SHA"' in text
    receipt, program = jobs["main-receipt"], jobs["program-trajectory"]
    assert receipt["env"]["SOURCE_SHA"] == program["env"]["SOURCE_SHA"]
    pr = str(jobs["pr-delta"])
    assert "--mode pr" in pr and "--mode receipt" not in pr and "--mode program" not in pr
    terminal = jobs["pr-delta"]["steps"][-1]
    assert terminal["if"] == "always()" and "steps.admission.outcome" in terminal["run"]
    assert "needs" not in receipt and "needs" not in program
    assert "--mode program" in str(program) and "continue-on-error" not in str(program)


for name in "exactly_one_active_contract_drift_workflow exact_three_live_cdg_check_names contract_drift_triggers_pr_main_schedule_dispatch contract_drift_pr_has_no_governed_path_filter_gap contract_drift_pr_uses_event_bound_full_shas contract_drift_non_pr_events_resolve_one_sha_for_receipt_and_program pull_request_invokes_only_pr_mode terminal_aggregator_fails_if_pr_delta_is_skipped_cancelled_or_missing main_receipt_job_is_distinct_and_successful_when_trajectory_is_red program_trajectory_preserves_real_red_exit".split():
    globals()[f"test_{name}"] = _contract
