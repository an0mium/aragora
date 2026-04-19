from __future__ import annotations

from pathlib import Path

from aragora.swarm.proof_first_queue import (
    TW01_ALLOW_FLAG_ENV,
    classify_proof_first_queue_issue,
)
from aragora.swarm.roadmap_priority import RoadmapPriorityPolicy


def test_classify_proof_first_queue_issue_allows_do_now_roadmap_lane() -> None:
    decision = classify_proof_first_queue_issue(
        "[CS-01..03] Reconcile docs/status surfaces to current proof",
        "Keep roadmap, status, and positioning docs narrower than measured proof.",
        roadmap_policy=RoadmapPriorityPolicy(
            do_now=frozenset({"CS-01", "CS-02", "CS-03"}),
            delay=frozenset({"BC-07"}),
            avoid=frozenset({"CS-04"}),
        ),
    )

    assert decision.allowed is True
    assert decision.lane == "roadmap_do_now"


def test_classify_proof_first_queue_issue_allows_tw02_benchmark_follow_up() -> None:
    decision = classify_proof_first_queue_issue(
        "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
        "Refresh benchmark corpus freshness after stale closed issues were detected in the truth artifact.",
    )

    assert decision.allowed is True
    assert decision.lane == "benchmark_regression"


def test_classify_proof_first_queue_issue_allows_tw03_rescue_follow_up() -> None:
    decision = classify_proof_first_queue_issue(
        "[TW-03] Productize repeated rescue class: blocked-auth-failure",
        "Repeated rescue class productization follow-up for rescue ledger drift.",
    )

    assert decision.allowed is True
    assert decision.lane == "rescue_productization"


def test_classify_proof_first_queue_issue_blocks_generic_cleanup() -> None:
    decision = classify_proof_first_queue_issue(
        "Replace silent exception swallowing in postgres_store.py",
        "Tighten exception hygiene in a single module.",
    )

    assert decision.allowed is False
    assert decision.lane == "non_canonical"


# ---------------------------------------------------------------------------
# TW-01 test-authoring expansion (flagged, default off)
# ---------------------------------------------------------------------------


FLAG_ON = {TW01_ALLOW_FLAG_ENV: "1"}
FLAG_OFF: dict[str, str] = {}


def _tw01_body(
    *,
    files: list[str] | None = None,
    validation: str = "pytest tests/foo/test_bar.py -q",
    extra_sections: str = "",
) -> str:
    files = files if files is not None else ["aragora/swarm/proof_first_queue.py"]
    file_lines = "\n".join(f"- `{path}`" for path in files)
    body = "## Task\n\nAuthor a new test module.\n\n"
    if files:
        body += f"## Files\n\n{file_lines}\n\n"
    body += f"## Validation\n\n```bash\n{validation}\n```\n"
    if extra_sections:
        body += "\n" + extra_sections
    return body


def test_tw01_flag_off_falls_through_to_non_canonical(tmp_path: Path) -> None:
    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests for foo",
        _tw01_body(),
        repo_root=tmp_path,
        env=FLAG_OFF,
    )

    assert decision.allowed is False
    assert decision.lane == "non_canonical"


def test_tw01_flag_on_accepts_well_formed_issue(tmp_path: Path) -> None:
    target = tmp_path / "aragora" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("# module\n", encoding="utf-8")

    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests for foo.py",
        _tw01_body(files=["aragora/foo.py"]),
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is True
    assert decision.lane == "test_authoring"
    assert "tw-01" in decision.matched_terms
    assert "aragora/foo.py" in decision.matched_terms


def test_tw01_flag_on_rejects_issue_missing_files_section(tmp_path: Path) -> None:
    body = (
        "## Task\n\nAuthor a new test module.\n\n"
        "## Validation\n\n```bash\npytest tests/foo/test_bar.py -q\n```\n"
    )
    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests",
        body,
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is False
    assert decision.lane == "tw01_rejected"
    assert "missing `## Files`" in decision.reason


def test_tw01_flag_on_rejects_issue_with_no_files_listed(tmp_path: Path) -> None:
    body = (
        "## Task\n\nAuthor a new test module.\n\n"
        "## Files\n\n(To be filled in later.)\n\n"
        "## Validation\n\n```bash\npytest -q\n```\n"
    )
    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests",
        body,
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is False
    assert decision.lane == "tw01_rejected"
    assert "no target files" in decision.reason


def test_tw01_flag_on_rejects_issue_missing_validation(tmp_path: Path) -> None:
    target = tmp_path / "aragora" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("# module\n", encoding="utf-8")

    body = "## Task\n\nAuthor a new test module.\n\n## Files\n\n- `aragora/foo.py`\n"
    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests for foo",
        body,
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is False
    assert decision.lane == "tw01_rejected"
    assert "missing `## Validation`" in decision.reason


def test_tw01_flag_on_rejects_issue_without_pytest_command(tmp_path: Path) -> None:
    target = tmp_path / "aragora" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("# module\n", encoding="utf-8")

    body = _tw01_body(
        files=["aragora/foo.py"],
        validation="make test-foo",
    )
    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests for foo",
        body,
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is False
    assert decision.lane == "tw01_rejected"
    assert "pytest" in decision.reason


def test_tw01_flag_on_rejects_issue_pointing_at_nonexistent_file(tmp_path: Path) -> None:
    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests for missing",
        _tw01_body(files=["aragora/does_not_exist.py"]),
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is False
    assert decision.lane == "tw01_rejected"
    assert "do not exist" in decision.reason


def test_tw01_flag_on_accepts_when_any_listed_file_exists(tmp_path: Path) -> None:
    target = tmp_path / "aragora" / "real.py"
    target.parent.mkdir(parents=True)
    target.write_text("# module\n", encoding="utf-8")

    decision = classify_proof_first_queue_issue(
        "[TW-01] Author regression tests",
        _tw01_body(files=["aragora/missing.py", "aragora/real.py"]),
        repo_root=tmp_path,
        env=FLAG_ON,
    )

    assert decision.allowed is True
    assert decision.lane == "test_authoring"
    assert "aragora/real.py" in decision.matched_terms


def test_tw02_behaviour_unchanged_by_flag(tmp_path: Path) -> None:
    # Regression test: TW-02 continues to classify as benchmark_regression
    # whether the TW-01 flag is on or off.
    for env in (FLAG_OFF, FLAG_ON):
        decision = classify_proof_first_queue_issue(
            "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
            (
                "Refresh benchmark corpus freshness after stale closed issues "
                "were detected in the truth artifact."
            ),
            repo_root=tmp_path,
            env=env,
        )
        assert decision.allowed is True, f"flag={env} broke TW-02 acceptance"
        assert decision.lane == "benchmark_regression"
