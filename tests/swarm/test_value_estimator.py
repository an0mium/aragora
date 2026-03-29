"""Tests for task value-per-cost estimation."""

from aragora.swarm.value_estimator import (
    OutcomeRecord,
    ValueEstimate,
    estimate_from_issue,
    rank_issues,
)


def test_estimate_bugfix_scores_higher_than_docs():
    bugfix = estimate_from_issue(
        issue_number=1,
        title="Fix receipt query endpoint",
        body="Receipts not queryable via GET /api/v1/receipts. pytest tests/server -x -q",
    )
    docs = estimate_from_issue(
        issue_number=2,
        title="Update README formatting",
        body="Fix typos in documentation",
    )
    assert bugfix.priority_score > docs.priority_score
    assert bugfix.expected_value > docs.expected_value


def test_estimate_with_test_commands_higher_success():
    with_tests = estimate_from_issue(
        issue_number=1,
        title="Add harvest CLI command",
        body="Files: swarm.py\nAcceptance: pytest tests/cli/test_swarm.py -x -q passes.",
    )
    without_tests = estimate_from_issue(
        issue_number=2,
        title="Add harvest CLI command",
        body="Make it work",
    )
    assert with_tests.p_success > without_tests.p_success


def test_estimate_frontend_higher_cost():
    frontend = estimate_from_issue(
        issue_number=1,
        title="Build React dashboard component",
        body="Create a new dashboard with tsx components and tailwind styling",
    )
    backend = estimate_from_issue(
        issue_number=2,
        title="Fix receipt persistence",
        body="Wire receipt query to correct store. pytest tests/gauntlet -x -q",
    )
    assert frontend.expected_minutes > backend.expected_minutes
    assert frontend.merge_difficulty > backend.merge_difficulty


def test_rank_issues_orders_by_score():
    issues = [
        {"number": 1, "title": "Update docs", "body": "Fix typos", "labels": []},
        {
            "number": 2,
            "title": "Fix crash in receipt store",
            "body": "pytest tests/gauntlet -x -q\nFiles: receipts.py",
            "labels": ["proof"],
        },
        {
            "number": 3,
            "title": "Build Chrome extension",
            "body": "New browser extension",
            "labels": [],
        },
    ]
    ranked = rank_issues(issues)
    numbers = [est.issue_number for est, _ in ranked]
    # Receipt fix should rank highest (bugfix + proof + has tests)
    assert numbers[0] == 2
    # Chrome extension should rank lowest (low success, external)
    assert numbers[-1] == 3


def test_compute_score_deterministic():
    est = ValueEstimate(
        issue_number=1,
        title="test",
        expected_value=0.8,
        p_success=0.7,
        proof_weight=0.9,
        unblock_weight=0.6,
        truthfulness=0.9,
    )
    score1 = est.compute_score()
    score2 = est.compute_score()
    assert score1 == score2
    assert score1 > 0


def test_calibration_adjusts_estimates():
    history = [
        OutcomeRecord(
            issue_number=i,
            predicted_score=0.5,
            predicted_p_success=0.5,
            did_merge=(i % 3 == 0),
            worker_status="completed" if i % 3 == 0 else "needs_human",
        )
        for i in range(10)
    ]
    calibrated = estimate_from_issue(
        issue_number=99,
        title="Fix test",
        body="pytest tests/ -x -q",
        historical_outcomes=history,
    )
    assert calibrated.estimation_method == "calibrated"
    # Historical rate is ~33%, should pull p_success down from heuristic
    assert calibrated.p_success < 0.65


def test_sparse_body_lowers_success():
    sparse = estimate_from_issue(issue_number=1, title="Do thing", body="")
    detailed = estimate_from_issue(
        issue_number=2,
        title="Do thing",
        body="Detailed description with specific files to change and acceptance criteria including pytest commands"
        * 5,
    )
    assert detailed.p_success > sparse.p_success
