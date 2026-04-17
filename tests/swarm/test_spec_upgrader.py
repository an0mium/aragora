"""Unit tests for SpecUpgrader."""

from __future__ import annotations

import pytest

from aragora.swarm.spec_upgrader import (
    SpecUpgraderUnavailable,
    UpgradeFailureContext,
    UpgradeResult,
)


def test_upgrade_failure_context_construction():
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Do the thing.",
        issue_title="[TW-02] Improve X",
        track_tag="TW-02",
    )
    assert ctx.missing_bounds == ["acceptance criterion", "file-scope hint"]
    assert ctx.prior_attempts == 0
    assert ctx.track_tag == "TW-02"


def test_upgrade_failure_context_frozen():
    ctx = UpgradeFailureContext(
        missing_bounds=[],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="",
        track_tag=None,
    )
    with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
        ctx.prior_attempts = 1  # type: ignore[misc]


def test_upgrade_result_upgraded_shape():
    from aragora.swarm.spec import SwarmSpec

    spec = SwarmSpec()
    res = UpgradeResult(
        status="upgraded",
        upgraded_spec=spec,
        audit_markdown="stub",
        attempt_count=1,
        upgrade_path="deterministic",
        failure_context=UpgradeFailureContext(
            missing_bounds=[],
            preflight_diff=None,
            prior_attempts=0,
            original_issue_body="",
            issue_title="",
            track_tag=None,
        ),
        unresolved_questions=[],
    )
    assert res.status == "upgraded"
    assert res.upgraded_spec is spec
    assert res.unresolved_questions == []


def test_upgrade_result_escalated_shape():
    res = UpgradeResult(
        status="escalated",
        upgraded_spec=None,
        audit_markdown="stub",
        attempt_count=2,
        upgrade_path="deterministic+llm",
        failure_context=UpgradeFailureContext(
            missing_bounds=["acceptance criterion"],
            preflight_diff=None,
            prior_attempts=2,
            original_issue_body="",
            issue_title="",
            track_tag=None,
        ),
        unresolved_questions=["What is the acceptance criterion?"],
    )
    assert res.status == "escalated"
    assert res.upgraded_spec is None
    assert len(res.unresolved_questions) == 1


def test_spec_upgrader_unavailable_is_exception():
    with pytest.raises(SpecUpgraderUnavailable):
        raise SpecUpgraderUnavailable("LLM client timed out")


from aragora.swarm.spec_upgrader import _classify_missing_bounds


def test_classify_missing_bounds_all_categories():
    bounds = [
        "acceptance criterion",
        "file-scope hint",
        "constraint",
        "explicit work order",
    ]
    result = _classify_missing_bounds(bounds)
    assert result == {
        "needs_acceptance": True,
        "needs_file_scope": True,
        "needs_constraint": True,
        "needs_work_order": True,
    }


def test_classify_missing_bounds_partial():
    bounds = ["acceptance criterion"]
    result = _classify_missing_bounds(bounds)
    assert result["needs_acceptance"] is True
    assert result["needs_file_scope"] is False


def test_classify_missing_bounds_empty():
    result = _classify_missing_bounds([])
    assert all(v is False for v in result.values())


from pathlib import Path

from aragora.swarm.spec_upgrader import _extract_file_paths


def test_extract_file_paths_from_body(tmp_path, monkeypatch):
    # Create fake repo files
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    (tmp_path / "aragora" / "swarm" / "spec.py").write_text("")
    monkeypatch.chdir(tmp_path)

    body = (
        "Fix the thing in `aragora/swarm/boss_loop.py` and also "
        "the parser at aragora/swarm/spec.py. This imaginary/path.py does not exist."
    )
    paths = _extract_file_paths(body, repo_root=Path(tmp_path))
    assert "aragora/swarm/boss_loop.py" in paths
    assert "aragora/swarm/spec.py" in paths
    assert "imaginary/path.py" not in paths


def test_extract_file_paths_empty_body(tmp_path):
    assert _extract_file_paths("", repo_root=Path(tmp_path)) == []


def test_extract_file_paths_no_matches(tmp_path):
    body = "This issue has no file references, just prose."
    assert _extract_file_paths(body, repo_root=Path(tmp_path)) == []


from aragora.swarm.spec_upgrader import _infer_track_scope


def test_infer_track_scope_tw_validates_repo(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "__init__.py").write_text("")

    hints = _infer_track_scope(
        "TW-02", issue_body="refactor boss_loop logic", repo_root=Path(tmp_path)
    )
    assert hints == ["aragora/swarm/"]


def test_infer_track_scope_unknown_tag_returns_empty(tmp_path):
    hints = _infer_track_scope("XYZ-99", issue_body="", repo_root=Path(tmp_path))
    assert hints == []


def test_infer_track_scope_design_heavy_returns_empty(tmp_path):
    # AGT-*/DIC-* are vision-layer; must not guess paths
    assert _infer_track_scope("AGT-01", issue_body="", repo_root=Path(tmp_path)) == []
    assert _infer_track_scope("DIC-15", issue_body="", repo_root=Path(tmp_path)) == []


def test_infer_track_scope_missing_directory_drops_hint(tmp_path):
    # Repo doesn't have aragora/swarm/ - hint is not validated, returns empty
    hints = _infer_track_scope("TW-02", issue_body="", repo_root=Path(tmp_path))
    assert hints == []
