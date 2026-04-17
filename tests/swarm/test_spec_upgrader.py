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


from aragora.swarm.spec_upgrader import _parse_audit_marker


def test_parse_audit_marker_valid():
    comment = "<!-- spec-upgraded:v1 attempt=1 -->\n\n## Upgrade audit\nblah blah"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 1
    assert valid is True


def test_parse_audit_marker_attempt_2():
    comment = "<!-- spec-upgraded:v1 attempt=2 -->\ncontent"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 2
    assert valid is True


def test_parse_audit_marker_corrupted_returns_max():
    comment = "<!-- spec-upgraded:v1 attempt=garbage -->\ncontent"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 2  # max_attempts sentinel
    assert valid is False


def test_parse_audit_marker_wrong_version():
    comment = "<!-- spec-upgraded:v2 attempt=1 -->\ncontent"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 2
    assert valid is False


def test_parse_audit_marker_no_marker():
    comment = "Some unrelated comment"
    attempt, valid = _parse_audit_marker(comment)
    assert attempt == 0
    assert valid is True


from unittest.mock import patch

from aragora.swarm.spec_upgrader import AuditPersistence


def test_audit_read_attempt_count_no_prior_marker():
    ap = AuditPersistence(issue_number=5898)
    with patch.object(
        ap,
        "_gh_list_comments",
        return_value=[
            {"id": 1, "body": "unrelated"},
            {"id": 2, "body": "also unrelated"},
        ],
    ):
        count, valid = ap.read_attempt_count()
    assert count == 0
    assert valid is True


def test_audit_read_attempt_count_existing_marker():
    ap = AuditPersistence(issue_number=5898)
    with patch.object(
        ap,
        "_gh_list_comments",
        return_value=[
            {"id": 1, "body": "<!-- spec-upgraded:v1 attempt=1 -->\n## Upgrade audit"},
        ],
    ):
        count, valid = ap.read_attempt_count()
    assert count == 1
    assert valid is True


def test_audit_upsert_creates_when_missing():
    ap = AuditPersistence(issue_number=5898)
    with (
        patch.object(ap, "_gh_list_comments", return_value=[]),
        patch.object(ap, "_gh_create_comment") as cc,
        patch.object(ap, "_gh_update_comment") as uc,
    ):
        ap.upsert(attempt=1, audit_markdown="## body")
        cc.assert_called_once()
        uc.assert_not_called()
        posted_body = cc.call_args.kwargs.get("body")
        assert posted_body is not None
        assert "<!-- spec-upgraded:v1 attempt=1 -->" in posted_body


def test_audit_upsert_updates_when_present():
    existing_comment = {
        "id": 42,
        "body": "<!-- spec-upgraded:v1 attempt=1 -->\nold",
    }
    ap = AuditPersistence(issue_number=5898)
    with (
        patch.object(ap, "_gh_list_comments", return_value=[existing_comment]),
        patch.object(ap, "_gh_create_comment") as cc,
        patch.object(ap, "_gh_update_comment") as uc,
    ):
        ap.upsert(attempt=2, audit_markdown="## new body")
        uc.assert_called_once()
        cc.assert_not_called()
        kwargs = uc.call_args.kwargs
        assert kwargs.get("comment_id") == 42
        assert "attempt=2" in (kwargs.get("body") or "")


import subprocess

from aragora.swarm.spec_upgrader import Escalator


def test_escalator_success():
    esc = Escalator(issue_number=5898)
    with (
        patch.object(esc, "_gh_add_label") as al,
        patch.object(esc, "_gh_create_comment") as cc,
    ):
        success = esc.escalate(
            unresolved_questions=["What file scope?", "What acceptance criterion?"],
            failure_context_summary="Missing all bounds",
        )
    assert success is True
    al.assert_called_once()
    cc.assert_called_once()


def test_escalator_label_failure_is_fail_closed():
    esc = Escalator(issue_number=5898)
    with (
        patch.object(esc, "_gh_add_label", side_effect=subprocess.CalledProcessError(1, "gh")),
        patch.object(esc, "_gh_create_comment") as cc,
    ):
        success = esc.escalate(
            unresolved_questions=["Q1"],
            failure_context_summary="summary",
        )
    assert success is False
    cc.assert_not_called()


def test_escalator_comment_failure_is_fail_closed():
    esc = Escalator(issue_number=5898)
    with (
        patch.object(esc, "_gh_add_label"),
        patch.object(esc, "_gh_create_comment", side_effect=subprocess.CalledProcessError(1, "gh")),
    ):
        success = esc.escalate(
            unresolved_questions=["Q1"],
            failure_context_summary="summary",
        )
    assert success is False


import json

from aragora.swarm.spec_upgrader import emit_upgrade_telemetry


def test_emit_upgrade_telemetry_writes_jsonl(tmp_path):
    metrics_path = tmp_path / "boss_metrics.jsonl"
    upgrade_id = emit_upgrade_telemetry(
        metrics_path=metrics_path,
        issue_number=5898,
        seam="A",
        attempt_count=1,
        status="upgraded",
        upgrade_path="deterministic",
        wall_clock_ms=432,
        audit_failed=False,
        escalation_failed=False,
        llm_tokens_in=0,
        llm_tokens_out=0,
        failure_reasons=["acceptance criterion"],
    )
    assert metrics_path.exists()
    line = metrics_path.read_text().strip()
    record = json.loads(line)
    assert record["event"] == "spec_upgrade"
    assert record["upgrade_id"] == upgrade_id
    assert record["issue_number"] == 5898
    assert record["seam"] == "A"
    assert record["status"] == "upgraded"


def test_emit_upgrade_telemetry_appends(tmp_path):
    metrics_path = tmp_path / "boss_metrics.jsonl"
    emit_upgrade_telemetry(
        metrics_path=metrics_path,
        issue_number=1,
        seam="A",
        attempt_count=1,
        status="upgraded",
        upgrade_path="deterministic",
        wall_clock_ms=1,
        audit_failed=False,
        escalation_failed=False,
        llm_tokens_in=0,
        llm_tokens_out=0,
        failure_reasons=[],
    )
    emit_upgrade_telemetry(
        metrics_path=metrics_path,
        issue_number=2,
        seam="B",
        attempt_count=2,
        status="escalated",
        upgrade_path="deterministic+llm",
        wall_clock_ms=2,
        audit_failed=False,
        escalation_failed=False,
        llm_tokens_in=10,
        llm_tokens_out=20,
        failure_reasons=["constraint"],
    )
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(lin) for lin in lines]
    assert recs[0]["issue_number"] == 1
    assert recs[1]["issue_number"] == 2


from aragora.swarm.spec_upgrader import MAX_ATTEMPTS, upgrade_spec


def test_upgrade_spec_tier1_success(tmp_path, monkeypatch):
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("")
    metrics = tmp_path / "boss_metrics.jsonl"

    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion", "file-scope hint"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="Fix `aragora/swarm/boss_loop.py` behaviour.",
        issue_title="[TW-02] Fix boss loop",
        track_tag="TW-02",
    )

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        AP.return_value.upsert.return_value = True
        result = upgrade_spec(
            spec,
            ctx,
            issue_number=5898,
            seam="A",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert result.status == "upgraded"
    assert result.attempt_count == 1
    assert result.upgrade_path == "deterministic"
    assert metrics.exists()


def test_upgrade_spec_escalates_on_max_attempts(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="",
        issue_title="[CS-01] stuck",
        track_tag="CS-01",
    )
    metrics = tmp_path / "boss_metrics.jsonl"

    with (
        patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP,
        patch("aragora.swarm.spec_upgrader.Escalator") as ESC,
    ):
        AP.return_value.read_attempt_count.return_value = (MAX_ATTEMPTS, True)
        ESC.return_value.escalate.return_value = True
        result = upgrade_spec(
            spec,
            ctx,
            issue_number=5903,
            seam="A",
            repo_root=Path(tmp_path),
            metrics_path=metrics,
            llm_client=None,
        )

    assert result.status == "escalated"
    ESC.return_value.escalate.assert_called_once()


def test_upgrade_spec_llm_unavailable_bubbles(tmp_path):
    spec = _make_unbounded_spec()
    ctx = UpgradeFailureContext(
        missing_bounds=["acceptance criterion"],
        preflight_diff=None,
        prior_attempts=0,
        original_issue_body="ambiguous",
        issue_title="[CS-01] x",
        track_tag="CS-01",
    )
    metrics = tmp_path / "boss_metrics.jsonl"
    mock_client = MagicMock()
    mock_client.complete.side_effect = ConnectionError("timeout")

    with patch("aragora.swarm.spec_upgrader.AuditPersistence") as AP:
        AP.return_value.read_attempt_count.return_value = (0, True)
        with pytest.raises(SpecUpgraderUnavailable):
            upgrade_spec(
                spec,
                ctx,
                issue_number=5903,
                seam="A",
                repo_root=Path(tmp_path),
                metrics_path=metrics,
                llm_client=mock_client,
            )
