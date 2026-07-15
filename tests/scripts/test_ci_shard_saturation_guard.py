"""Tests for scripts/ci_shard_saturation_guard.py (pure analysis helpers)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci_shard_saturation_guard.py"
_spec = importlib.util.spec_from_file_location("ci_shard_saturation_guard", _SCRIPT)
guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = guard
_spec.loader.exec_module(guard)


def _job(
    name: str = "test-fast (debate-am, debate-am, debate, 30)",
    started: str = "2026-07-15T16:00:00Z",
    completed: str = "2026-07-15T16:22:00Z",
    run_step_conclusion: str | None = "success",
) -> dict:
    return {
        "name": name,
        "started_at": started,
        "completed_at": completed,
        "conclusion": "success",
        "steps": [
            {"name": "Resolve shard relevance", "conclusion": "success"},
            {"name": "Checkout", "conclusion": "success"},
            {"name": "Run debate-am tests", "conclusion": run_step_conclusion},
        ],
    }


class TestParseShardName:
    def test_extracts_first_token(self):
        assert guard.parse_shard_name("test-fast (debate-am, debate-am, debate, 30)") == "debate-am"

    def test_handles_truncated_job_name(self):
        # GitHub truncates long matrix names with a trailing "..."
        name = "test-fast (infra, tests/nomic tests/control_plane tests/rbac tes..."
        assert guard.parse_shard_name(name) == "infra"

    def test_non_test_fast_jobs_ignored(self):
        assert guard.parse_shard_name("lint") is None
        assert guard.parse_shard_name("test (ubuntu-latest, 3.12)") is None
        assert guard.parse_shard_name("test-fast-summary") is None


class TestJobExecuted:
    def test_executed_when_run_step_succeeded(self):
        assert guard.job_executed(_job(run_step_conclusion="success"))

    def test_cap_killed_job_counts_as_executed(self):
        assert guard.job_executed(_job(run_step_conclusion="cancelled"))

    def test_path_filtered_job_excluded(self):
        assert not guard.job_executed(_job(run_step_conclusion="skipped"))

    def test_job_without_steps_excluded(self):
        assert not guard.job_executed({"name": "test-fast (agents, ...)", "steps": None})


class TestJobDurationMinutes:
    def test_computes_minutes(self):
        assert guard.job_duration_minutes(_job()) == pytest.approx(22.0)

    def test_missing_timestamps(self):
        assert guard.job_duration_minutes({"started_at": None, "completed_at": None}) is None

    def test_negative_duration_rejected(self):
        job = _job(started="2026-07-15T17:00:00Z", completed="2026-07-15T16:00:00Z")
        assert guard.job_duration_minutes(job) is None


class TestPercentile:
    def test_p95_nearest_rank(self):
        values = [float(v) for v in range(1, 101)]
        assert guard.percentile(values, 95) == 95.0

    def test_small_sample(self):
        assert guard.percentile([10.0, 20.0, 30.0], 95) == 30.0
        assert guard.percentile([10.0], 95) == 10.0

    def test_p50(self):
        assert guard.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            guard.percentile([], 95)


class TestCollectShardDurations:
    def test_groups_by_shard_and_filters(self):
        jobs = [
            _job(),
            _job(completed="2026-07-15T16:10:00Z"),
            _job(name="test-fast (agents, tests/agents, agents, 30)"),
            _job(run_step_conclusion="skipped"),  # path-filtered: excluded
            {"name": "lint", "steps": []},  # non-shard job: excluded
        ]
        durations = guard.collect_shard_durations(jobs)
        assert sorted(durations) == ["agents", "debate-am"]
        assert len(durations["debate-am"]) == 2


class TestLoadActiveShards:
    _WORKFLOW = """\
jobs:
  lint:
    steps:
      - name: not-a-shard
  test-fast:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        category:
          - name: agents
            pytest_args: tests/agents
            scope: agents
            timeout: 30
          - name: debate-1
            pytest_args: ""
            resolver: debate-1
            scope: debate
            timeout: 30
    steps:
      - name: Checkout
        uses: actions/checkout@v4
  test-summary:
    steps:
      - name: also-not-a-shard
"""

    def test_parses_matrix_shard_names(self, tmp_path):
        wf = tmp_path / "test.yml"
        wf.write_text(self._WORKFLOW)
        assert guard.load_active_shards(wf) == frozenset({"agents", "debate-1"})

    def test_missing_file_fails_open(self, tmp_path):
        assert guard.load_active_shards(tmp_path / "nope.yml") is None

    def test_unrecognized_structure_fails_open(self, tmp_path):
        wf = tmp_path / "test.yml"
        wf.write_text("jobs:\n  other-job:\n    steps: []\n")
        assert guard.load_active_shards(wf) is None

    def test_parses_real_repo_workflow(self):
        shards = guard.load_active_shards(_SCRIPT.parents[1] / ".github" / "workflows" / "test.yml")
        assert shards is not None
        assert "agents" in shards
        assert "infra" in shards
        # No step names should leak in.
        assert "Checkout" not in shards


class TestAnalyze:
    def test_breach_and_cap_hits(self):
        durations = {
            "debate-am": [22.0, 23.0, 24.0, 29.5],
            "agents": [5.0, 6.0, 7.0],
        }
        stats = guard.analyze(durations, threshold_minutes=20.0, cap_minutes=30.0)
        assert [s.shard for s in stats] == ["debate-am", "agents"]  # hottest first
        debate = stats[0]
        assert debate.breach
        assert debate.cap_hits == 1  # 29.5 >= cap - 1.0
        assert not stats[1].breach

    def test_no_breach_at_threshold(self):
        stats = guard.analyze({"agents": [20.0]}, threshold_minutes=20.0, cap_minutes=30.0)
        assert not stats[0].breach

    def test_retired_shard_reported_but_never_breaches(self):
        durations = {"debate-am": [29.0] * 5, "debate-1": [25.0] * 5}
        stats = guard.analyze(
            durations, 20.0, 30.0, active_shards=frozenset({"debate-1", "agents"})
        )
        by_shard = {s.shard: s for s in stats}
        assert not by_shard["debate-am"].breach
        assert not by_shard["debate-am"].active
        assert by_shard["debate-1"].breach

    def test_unknown_layout_fails_open(self):
        stats = guard.analyze({"debate-am": [29.0] * 5}, 20.0, 30.0, active_shards=None)
        assert stats[0].breach and stats[0].active


class TestEmitOutputs:
    def test_writes_annotation_summary_and_outputs(self, tmp_path, capsys, monkeypatch):
        summary = tmp_path / "summary.md"
        output = tmp_path / "output.txt"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))

        stats = guard.analyze(
            {"debate-am": [25.0] * 5, "agents": [5.0] * 5},
            threshold_minutes=20.0,
            cap_minutes=30.0,
        )
        guard.emit_outputs(stats, 20.0, 30.0, runs_analyzed=5, days=14)

        stdout = capsys.readouterr().out
        assert "::warning title=CI shard saturation risk::Shard 'debate-am'" in stdout
        assert "ci_resolve_test_shard.py" in stdout

        assert "| debate-am |" in summary.read_text()
        out_text = output.read_text()
        assert "breach=1" in out_text
        assert "breach_shards=debate-am" in out_text
        assert "CI_SHARD_REPORT_EOF" in out_text

    def test_retired_breach_is_informational_only(self, tmp_path, capsys, monkeypatch):
        output = tmp_path / "output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

        stats = guard.analyze(
            {"debate-am": [29.0] * 5},
            threshold_minutes=20.0,
            cap_minutes=30.0,
            active_shards=frozenset({"debate-1"}),
        )
        guard.emit_outputs(stats, 20.0, 30.0, runs_analyzed=5, days=14)

        stdout = capsys.readouterr().out
        assert "::warning" not in stdout
        assert "retired (aging out)" in stdout
        assert "breach=0" in output.read_text()

    def test_clean_run_reports_no_breach(self, tmp_path, capsys, monkeypatch):
        output = tmp_path / "output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

        stats = guard.analyze({"agents": [5.0] * 5}, 20.0, 30.0)
        guard.emit_outputs(stats, 20.0, 30.0, runs_analyzed=5, days=14)

        assert "::warning" not in capsys.readouterr().out
        assert "breach=0" in output.read_text()
