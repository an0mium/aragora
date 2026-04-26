"""Tests for :mod:`scripts.replay_mode3_sample` — rubric replay helpers.

Scoped to the pure-function helpers: severity classifier, severity gate,
and advocate-rebalance logic. Full-disk end-to-end replay is exercised
implicitly by the manual ``scripts/replay_mode3_sample.py`` invocation
captured in ``docs/status/2026-04-24-mode3-rc1-calibration-post-fix.md``.

Mission context: epic #6505, fix #4 (re-derive precision on the 15-brief
sample through the new post-#6510/#6514 rubric without new API spend).
These tests lock in the rubric contract so future prompt/verdict
changes can't silently regress the replay output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "replay_mode3_sample.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "aragora_tests._replay_mode3_sample",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def cli():
    return _load_cli_module()


# --- Severity classifier --------------------------------------------------


class TestClassifySeverity:
    def test_blocker_language_returns_high(self, cli):
        text = "This is a hard blocker on merge-critical paths."
        assert cli._classify_severity(text) == "high"

    def test_security_vulnerability_returns_high(self, cli):
        text = "Introduces a security vulnerability via SSRF."
        assert cli._classify_severity(text) == "high"

    def test_should_be_addressed_returns_medium(self, cli):
        text = "These findings should be addressed before merging."
        assert cli._classify_severity(text) == "medium"

    def test_technical_debt_returns_medium(self, cli):
        text = "Introduces hidden coupling and technical debt."
        assert cli._classify_severity(text) == "medium"

    def test_minor_returns_low(self, cli):
        text = "The security implementation is sound with minor hardening opportunities."
        assert cli._classify_severity(text) == "low"

    def test_empty_text_defaults_to_low(self, cli):
        # Conservative-by-default: no tier match → low, not dropped.
        assert cli._classify_severity("") == "low"

    def test_precedence_high_beats_medium(self, cli):
        # Single text matching both tiers → highest wins.
        text = "Must not merge: technical debt makes this a hard blocker."
        assert cli._classify_severity(text) == "high"


# --- Severity gate --------------------------------------------------------


class TestApplySeverityGate:
    def test_approve_candidate_passes_through(self, cli):
        assert cli.apply_severity_gate("approve_candidate", {"high": 0}) == "approve_candidate"

    def test_needs_human_attention_passes_through(self, cli):
        assert (
            cli.apply_severity_gate("needs_human_attention", {"high": 5}) == "needs_human_attention"
        )

    def test_repair_first_with_high_stays(self, cli):
        assert (
            cli.apply_severity_gate("repair_first", {"high": 1, "medium": 0, "low": 0})
            == "repair_first"
        )

    def test_repair_first_without_high_downgrades(self, cli):
        assert (
            cli.apply_severity_gate("repair_first", {"high": 0, "medium": 3, "low": 2})
            == "approve_with_followups"
        )

    def test_missing_high_key_treated_as_zero(self, cli):
        # Contract: ``severity_counts.get("high", 0)`` — an absent key
        # behaves the same as ``0`` so malformed callers downgrade rather
        # than leak a spurious repair_first.
        assert cli.apply_severity_gate("repair_first", {"medium": 2}) == "approve_with_followups"


# --- Advocate rebalance ---------------------------------------------------


class TestApplyAdvocateRebalance:
    def test_approve_candidate_never_downgrades(self, cli):
        # Monotone invariant: advocate never pushes toward blocking.
        assert (
            cli.apply_advocate_rebalance(
                "approve_candidate",
                panel_weight_against_approve=0.9,
                advocate_confidence=0.1,
            )
            == "approve_candidate"
        )

    def test_strong_advocate_flips_repair_first(self, cli):
        assert (
            cli.apply_advocate_rebalance(
                "repair_first",
                panel_weight_against_approve=0.5,
                advocate_confidence=0.7,
            )
            == "approve_candidate"
        )

    def test_weak_advocate_preserves_repair_first(self, cli):
        assert (
            cli.apply_advocate_rebalance(
                "repair_first",
                panel_weight_against_approve=0.8,
                advocate_confidence=0.2,
            )
            == "repair_first"
        )

    def test_advocate_flips_followups_to_approve(self, cli):
        assert (
            cli.apply_advocate_rebalance(
                "approve_with_followups",
                panel_weight_against_approve=0.4,
                advocate_confidence=0.5,
            )
            == "approve_candidate"
        )

    def test_advocate_breaks_needs_human_tie(self, cli):
        # Ties escalate to needs_human_attention; a confident advocate
        # unwedges them toward approve.
        assert (
            cli.apply_advocate_rebalance(
                "needs_human_attention",
                panel_weight_against_approve=0.6,
                advocate_confidence=0.7,
            )
            == "approve_candidate"
        )


# --- Advocate confidence heuristic ---------------------------------------


class TestAdvocateConfidence:
    def test_positive_evidence_raises_confidence(self, cli):
        summary = (
            "The PR reports a passing pytest run across its four targeted test "
            "modules and clean ruff output."
        )
        assert cli._advocate_confidence(summary) > 0.0

    def test_penalties_reduce_confidence(self, cli):
        # Positive hits plus penalties should net below the raw positive score.
        summary = (
            "Passing tests exercise happy-path unit behavior; however, "
            "no CI artifact links, evidence strength is low, untested "
            "multi-process scenarios remain."
        )
        # The calibration brief pattern — surface-passing tests + penalties —
        # should produce a low-but-nonnegative advocate confidence.
        conf = cli._advocate_confidence(summary)
        assert 0.0 <= conf <= 0.5

    def test_no_evidence_is_zero(self, cli):
        assert cli._advocate_confidence("") == 0.0


# --- Label override sidecar ----------------------------------------------


class TestLoadLabelOverride:
    def test_returns_none_when_sidecar_missing(self, cli, tmp_path):
        brief = tmp_path / "pr-1-abc.json"
        brief.write_text(json.dumps({"pr_number": 1}))
        assert cli._load_label_override(brief) is None

    def test_loads_sidecar_when_present(self, cli, tmp_path):
        brief = tmp_path / "pr-1-abc.json"
        brief.write_text(json.dumps({"pr_number": 1}))
        sidecar = brief.with_suffix(".severity.json")
        sidecar.write_text(json.dumps({"high": 2, "medium": 3, "low": 4}))
        override = cli._load_label_override(brief)
        assert override == {"high": 2, "medium": 3, "low": 4}

    def test_sidecar_defaults_missing_keys_to_zero(self, cli, tmp_path):
        brief = tmp_path / "pr-1-abc.json"
        brief.write_text(json.dumps({"pr_number": 1}))
        sidecar = brief.with_suffix(".severity.json")
        sidecar.write_text(json.dumps({"high": 1}))
        override = cli._load_label_override(brief)
        assert override == {"high": 1, "medium": 0, "low": 0}


# --- End-to-end replay_brief ---------------------------------------------


class TestReplayBrief:
    def _write_brief(self, tmp_path, **overrides):
        brief = {
            "pr_number": 1234,
            "head_sha": "abc123def456",
            "recommendation": "repair_first",
            "overall_confidence": 0.8,
            "validation_summary": "Passing tests. Clean ruff.",
            "role_findings": [
                {
                    "agent": "claude_core",
                    "finding_text": "The implementation is sound with minor opportunities.",
                    "confidence": 0.85,
                    "role": "logic_reviewer",
                },
                {
                    "agent": "gpt_core",
                    "finding_text": "Security is fine; minor hardening possible.",
                    "confidence": 0.85,
                    "role": "security_reviewer",
                },
            ],
        }
        brief.update(overrides)
        path = tmp_path / f"pr-{brief['pr_number']}-abc123.json"
        path.write_text(json.dumps(brief))
        return path

    def test_all_low_severity_downgrades_repair_first(self, cli, tmp_path):
        path = self._write_brief(tmp_path)
        result = cli.replay_brief(path)
        assert result.old_verdict == "repair_first"
        assert result.severity_counts["high"] == 0
        assert result.new_verdict_sev_gate == "approve_with_followups"

    def test_high_severity_finding_preserves_repair_first(self, cli, tmp_path):
        path = self._write_brief(
            tmp_path,
            role_findings=[
                {
                    "agent": "claude_core",
                    "finding_text": "This is a hard blocker — security vulnerability.",
                    "confidence": 0.9,
                    "role": "logic_reviewer",
                },
            ],
        )
        result = cli.replay_brief(path)
        assert result.severity_counts["high"] >= 1
        assert result.new_verdict_sev_gate == "repair_first"

    def test_manual_override_wins_over_heuristic(self, cli, tmp_path):
        path = self._write_brief(tmp_path)
        sidecar = path.with_suffix(".severity.json")
        sidecar.write_text(json.dumps({"high": 3, "medium": 0, "low": 0}))
        result = cli.replay_brief(path)
        assert result.notes == "severity=manual"
        assert result.severity_counts == {"high": 3, "medium": 0, "low": 0}
        assert result.new_verdict_sev_gate == "repair_first"


class TestMain:
    def test_missing_default_briefs_dir_explains_local_archive_dependency(
        self,
        cli,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        missing_repo = tmp_path / "clean-checkout"
        default_briefs = missing_repo / ".aragora" / "review-queue" / "briefs"
        monkeypatch.setattr(cli, "BRIEFS_DIR", default_briefs)

        rc = cli.main([])

        captured = capsys.readouterr()
        assert rc == 2
        assert "briefs directory not found" in captured.err
        assert ".aragora/ is intentionally gitignored" in captured.err
        assert "--briefs-dir" in captured.err
