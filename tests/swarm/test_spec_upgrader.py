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


from aragora.swarm.spec_upgrader import _drift_to_acceptance_criterion


def test_drift_files_mismatch_generates_scoping_criterion():
    drift = {
        "expected": {"files": ["aragora/swarm/a.py"]},
        "actual": {"files": ["aragora/swarm/a.py", "unrelated/b.py"]},
    }
    crit = _drift_to_acceptance_criterion(drift)
    assert crit is not None
    assert "aragora/swarm/a.py" in crit
    assert "unrelated/b.py" not in crit  # Don't name disallowed paths positively
    assert "scope" in crit.lower() or "restrict" in crit.lower()


def test_drift_none_returns_none():
    assert _drift_to_acceptance_criterion(None) is None


def test_drift_identical_returns_none():
    drift = {"expected": {"files": ["a"]}, "actual": {"files": ["a"]}}
    assert _drift_to_acceptance_criterion(drift) is None


from aragora.swarm.spec import SwarmSpec
from aragora.swarm.spec_upgrader import _tier1_enrich


def _make_unbounded_spec():
    """Build a minimally-underspecified SwarmSpec for testing."""
    return SwarmSpec(
        raw_goal="Improve boss_loop",
        refined_goal="Improve boss_loop",
        acceptance_criteria=[],
        constraints=[],
        file_scope_hints=[],
        work_orders=[],
    )


def test_tier1_enriches_from_body_and_track_tag(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    (tmp_path / "aragora" / "swarm" / "__init__.py").write_text("")

    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Fix bugs in `aragora/swarm/boss_loop.py`.",
        issue_title="[TW-02] Fix boss loop bugs",
        track_tag="TW-02",
    )
    upgraded = _tier1_enrich(spec, ctx, repo_root=Path(tmp_path))
    assert upgraded is not None
    assert "aragora/swarm/boss_loop.py" in upgraded.file_scope_hints
    assert upgraded.acceptance_criteria  # non-empty after enrichment


def test_tier1_returns_none_when_cannot_bound(tmp_path):
    # No body content, no track tag scope (AGT is design-heavy), no drift
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=[
            "acceptance criterion",
            "file-scope hint",
            "constraint",
            "explicit work order",
        ],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="[AGT-01] Design-heavy ambiguous",
        track_tag="AGT-01",
    )
    result = _tier1_enrich(spec, ctx, repo_root=Path(tmp_path))
    assert result is None


from unittest.mock import MagicMock

from aragora.swarm.spec_upgrader import _tier2_enrich


def test_tier2_enrich_success(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Ambiguous task.",
        issue_title="[CS-01] Stuff",
        track_tag="CS-01",
    )
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        '{"acceptance_criteria": ["The code produces output matching docs/examples/X.md"], '
        '"file_scope_hints": ["aragora/swarm/boss_loop.py"], '
        '"constraints": ["No changes outside listed files"], '
        '"work_orders": [{"description": "Add regression test for X"}]}'
    )
    result = _tier2_enrich(spec, ctx, client=mock_client, repo_root=Path(tmp_path))
    assert result is not None
    assert result.acceptance_criteria


def test_tier2_enrich_malformed_json_raises(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="",
        track_tag=None,
    )
    mock_client = MagicMock()
    mock_client.complete.return_value = "this is not json"
    from aragora.swarm.spec_upgrader import _LLMLogicFailure

    with pytest.raises(_LLMLogicFailure):
        _tier2_enrich(spec, ctx, client=mock_client, repo_root=Path(tmp_path))


def test_tier2_enrich_transient_raises_unavailable(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="",
        track_tag=None,
    )
    mock_client = MagicMock()
    mock_client.complete.side_effect = ConnectionError("api 503")
    with pytest.raises(SpecUpgraderUnavailable):
        _tier2_enrich(spec, ctx, client=mock_client, repo_root=Path(tmp_path))
