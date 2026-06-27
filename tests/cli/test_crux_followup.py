"""Tests for the ``aragora crux-followup`` CLI command (DIC-17).

Exercises: flag-gating, input loading, threshold filtering, top-k limiting,
JSON/text output shapes, and the boss-ready label invariant.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# aragora.epistemic.__init__ and aragora.reasoning.__init__ eagerly import
# modules that need yaml (PyYAML) and pydantic, which are absent from the
# pytest virtualenv.  Provide lightweight stubs before any aragora import so
# the import chains resolve without affecting the code under test.
for _stub in ["yaml", "pydantic", "pydantic.fields", "pydantic_settings", "pydantic_settings.main"]:
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

from aragora.cli.commands.crux_followup import cmd_crux_followup  # noqa: E402
from aragora.reasoning.cruxset import Crux, CruxPosition, CruxSet  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cruxset(*, scores: list[float] | None = None) -> CruxSet:
    """Build a minimal CruxSet with cruxes at the given load_bearing_scores."""
    if scores is None:
        scores = [0.85, 0.70]
    cruxes = [
        Crux(
            crux_id=f"c{i}",
            statement=f"Statement for crux {i}",
            positions=(
                CruxPosition(side="for", agents=(f"agent_a{i}",)),
                CruxPosition(side="against", agents=(f"agent_b{i}",)),
            ),
            load_bearing_score=score,
            evidence_gaps=(f"gap_{i}",),
            counterfactual=f"Flipping c{i} changes the decision.",
            candidate_verifier="docs/spec.md",
        )
        for i, score in enumerate(scores)
    ]
    return CruxSet.build(
        question="Should we ship the feature?",
        cruxes=cruxes,
        decision="hold",
    )


def _args(**kwargs):
    """Build a minimal argparse.Namespace for cmd_crux_followup."""
    import argparse

    defaults = dict(
        cruxset_file=None,
        threshold=0.6,
        top_k=5,
        json=False,
        file_issues=False,
        repo="",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


class TestInputLoading:
    def test_bad_json_exits_2(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("not-json", encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f)))
        assert rc == 2
        captured = capsys.readouterr()
        assert "not JSON" in captured.err

    def test_not_a_cruxset_exits_2(self, tmp_path, capsys):
        f = tmp_path / "other.json"
        f.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f)))
        assert rc == 2
        captured = capsys.readouterr()
        assert "not a CruxSet" in captured.err

    def test_missing_file_exits_2(self, tmp_path, capsys):
        rc = cmd_crux_followup(_args(cruxset_file=str(tmp_path / "no_such_file.json")))
        assert rc == 2
        captured = capsys.readouterr()
        assert "does not exist" in captured.err


# ---------------------------------------------------------------------------
# Threshold and top-k filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_no_qualifying_cruxes_text(self, tmp_path, capsys):
        cs = _make_cruxset(scores=[0.5, 0.4])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), threshold=0.6))
        assert rc == 0
        captured = capsys.readouterr()
        assert "No qualifying" in captured.out

    def test_threshold_filters_out_low_scores(self, tmp_path, capsys):
        cs = _make_cruxset(scores=[0.85, 0.40])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), threshold=0.8))
        assert rc == 0
        captured = capsys.readouterr()
        assert "1 follow-up proposal" in captured.out

    def test_top_k_limits_to_one(self, tmp_path, capsys):
        cs = _make_cruxset(scores=[0.9, 0.88, 0.75])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), threshold=0.1, top_k=1))
        assert rc == 0
        captured = capsys.readouterr()
        assert "1 follow-up proposal" in captured.out


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------


class TestTextOutput:
    def test_qualifying_cruxes_show_title_and_rationale(self, tmp_path, capsys):
        cs = _make_cruxset(scores=[0.85])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f)))
        assert rc == 0
        captured = capsys.readouterr()
        assert "[DIC-17]" in captured.out
        assert "load_bearing_score" in captured.out or "rationale" in captured.out


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_structure_has_expected_keys(self, tmp_path, capsys):
        cs = _make_cruxset(scores=[0.85])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), json=True))
        assert rc == 0
        captured = capsys.readouterr()
        items = json.loads(captured.out)
        assert isinstance(items, list)
        assert len(items) == 1
        item = items[0]
        for key in ("source_key", "source_kind", "title", "rationale", "labels", "provenance"):
            assert key in item, f"missing key: {key}"

    def test_json_empty_list_when_no_qualifying(self, tmp_path, capsys):
        cs = _make_cruxset(scores=[0.1])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), threshold=0.9, json=True))
        assert rc == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == []


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------


class TestFlagGating:
    def test_flag_off_file_issues_exits_1(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", raising=False)
        cs = _make_cruxset()
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), file_issues=True))
        assert rc == 1

    def test_flag_off_error_names_the_flag(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", raising=False)
        cs = _make_cruxset()
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        cmd_crux_followup(_args(cruxset_file=str(f), file_issues=True))
        captured = capsys.readouterr()
        assert "ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED" in captured.err

    def test_flag_off_dry_run_succeeds(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", raising=False)
        cs = _make_cruxset()
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), file_issues=False))
        assert rc == 0

    def test_flag_on_file_issues_missing_repo_exits_1(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", "1")
        cs = _make_cruxset()
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), file_issues=True, repo=""))
        assert rc == 1
        captured = capsys.readouterr()
        assert "--repo" in captured.err

    def test_flag_on_file_issues_prints_commands_without_executing(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", "1")
        cs = _make_cruxset(scores=[0.95])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        rc = cmd_crux_followup(_args(cruxset_file=str(f), file_issues=True, repo="owner/repo"))
        assert rc == 0
        captured = capsys.readouterr()
        assert "would file 1 issue(s)" in captured.out
        assert "commands shown, not executed" in captured.out
        assert "gh issue create" in captured.out


# ---------------------------------------------------------------------------
# Queue-governance invariant
# ---------------------------------------------------------------------------


class TestQueueGovernance:
    def test_proposals_never_carry_boss_ready(self, tmp_path):
        cs = _make_cruxset(scores=[0.95, 0.90, 0.80])
        f = tmp_path / "cs.json"
        f.write_text(json.dumps(cs.to_json()), encoding="utf-8")
        import argparse

        # Call the proposal logic directly to verify label safety
        from aragora.epistemic.followup import propose_followup_for_cruxset

        proposals = propose_followup_for_cruxset(cs, top_k=10, load_bearing_threshold=0.1)
        for p in proposals:
            assert "boss-ready" not in p.labels, (
                f"proposal {p.source_key} has forbidden label 'boss-ready'"
            )
