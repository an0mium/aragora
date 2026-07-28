import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github/workflows/contract-drift-governance.yml").read_text()
DOC = yaml.load(TEXT, Loader=yaml.BaseLoader)
JOBS = DOC["jobs"]


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
