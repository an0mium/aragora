"""Tests for aragora/swarm/runbook_dispatcher.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aragora.swarm.runbook_dispatcher import (
    RunbookDirective,
    dispatch_runbook,
    resolve_runbook_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_runbook(path: Path, payload: object) -> Path:
    path.write_text(yaml.dump(payload), encoding="utf-8")
    return path


def _minimal_runbook(tmp_path: Path, name: str = "test_run") -> Path:
    return _write_runbook(
        tmp_path / "test_run.yaml",
        {
            "name": name,
            "directives": [
                {"target": "agent-1", "task": "implement feature X"},
            ],
        },
    )


# ---------------------------------------------------------------------------
# RunbookDirective
# ---------------------------------------------------------------------------


class TestRunbookDirective:
    def test_frozen(self):
        d = RunbookDirective(
            target="agent-1",
            task="fix bug",
            scope=["#123"],
            constraints=["no deps"],
            status="active",
        )
        with pytest.raises((AttributeError, TypeError)):
            d.target = "other"  # type: ignore[misc]

    def test_fields(self):
        d = RunbookDirective(
            target="t",
            task="k",
            scope=["a", "b"],
            constraints=["c"],
            status="pending",
        )
        assert d.target == "t"
        assert d.task == "k"
        assert d.scope == ["a", "b"]
        assert d.constraints == ["c"]
        assert d.status == "pending"


# ---------------------------------------------------------------------------
# resolve_runbook_path
# ---------------------------------------------------------------------------


class TestResolveRunbookPath:
    def test_absolute_yaml_path(self, tmp_path: Path):
        runbook_file = tmp_path / "my_runbook.yaml"
        runbook_file.touch()
        result = resolve_runbook_path(str(runbook_file), repo_root=tmp_path)
        assert result == runbook_file

    def test_relative_yaml_extension(self, tmp_path: Path):
        runbook_file = tmp_path / "sub" / "plan.yaml"
        runbook_file.parent.mkdir(parents=True)
        runbook_file.touch()
        result = resolve_runbook_path("sub/plan.yaml", repo_root=tmp_path)
        assert result == runbook_file.resolve()

    def test_relative_yml_extension(self, tmp_path: Path):
        runbook_file = tmp_path / "plan.yml"
        runbook_file.touch()
        result = resolve_runbook_path("plan.yml", repo_root=tmp_path)
        assert result == runbook_file.resolve()

    def test_name_only_local_aragora_path_exists(self, tmp_path: Path):
        local_dir = tmp_path / ".aragora" / "runbooks"
        local_dir.mkdir(parents=True)
        (local_dir / "sprint.yaml").touch()
        result = resolve_runbook_path("sprint", repo_root=tmp_path)
        assert result == (local_dir / "sprint.yaml").resolve()

    def test_name_only_falls_back_to_docs_path(self, tmp_path: Path):
        # .aragora/runbooks/missing.yaml does NOT exist — should fall back to docs
        result = resolve_runbook_path("missing", repo_root=tmp_path)
        expected = (tmp_path / "docs" / "runbooks" / "missing.yaml").resolve()
        assert result == expected

    def test_name_only_prefers_local_over_docs(self, tmp_path: Path):
        local_dir = tmp_path / ".aragora" / "runbooks"
        local_dir.mkdir(parents=True)
        (local_dir / "both.yaml").touch()
        docs_dir = tmp_path / "docs" / "runbooks"
        docs_dir.mkdir(parents=True)
        (docs_dir / "both.yaml").touch()
        result = resolve_runbook_path("both", repo_root=tmp_path)
        assert result == (local_dir / "both.yaml").resolve()


# ---------------------------------------------------------------------------
# dispatch_runbook — dry_run=True
# ---------------------------------------------------------------------------


class TestDispatchRunbookDryRun:
    def test_returns_expected_top_level_keys(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert set(result.keys()) == {"name", "path", "issued_by", "directives", "dry_run"}

    def test_dry_run_flag_is_true(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["dry_run"] is True

    def test_issued_by_propagated(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="boss-x", repo_root=tmp_path, dry_run=True)
        assert result["issued_by"] == "boss-x"

    def test_name_from_yaml_payload(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path, name="my_campaign")
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["name"] == "my_campaign"

    def test_name_falls_back_to_stem(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "unnamed.yaml",
            {"directives": [{"target": "a", "task": "do something"}]},
        )
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["name"] == "unnamed"

    def test_path_in_result(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["path"] == str(rb)

    def test_directives_list_returned(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "multi.yaml",
            {
                "name": "multi",
                "directives": [
                    {
                        "target": "agent-a",
                        "task": "task A",
                        "scope": ["#1"],
                        "constraints": ["no-x"],
                    },
                    {"target": "agent-b", "task": "task B"},
                ],
            },
        )
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        directives = result["directives"]
        assert len(directives) == 2
        assert directives[0]["target"] == "agent-a"
        assert directives[0]["task"] == "task A"
        assert directives[0]["scope"] == ["#1"]
        assert directives[0]["constraints"] == ["no-x"]
        assert directives[1]["target"] == "agent-b"

    def test_dry_run_does_not_call_set_assignment(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        with patch("aragora.swarm.session_coordinator.set_assignment") as mock_sa:
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        mock_sa.assert_not_called()

    def test_default_status_applied(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "ds.yaml",
            {
                "default_status": "review",
                "directives": [{"target": "x", "task": "y"}],
            },
        )
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["status"] == "review"

    def test_directive_status_overrides_default(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "ov.yaml",
            {
                "default_status": "review",
                "directives": [{"target": "x", "task": "y", "status": "blocked"}],
            },
        )
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["status"] == "blocked"


# ---------------------------------------------------------------------------
# dispatch_runbook — dry_run=False (set_assignment called)
# ---------------------------------------------------------------------------


class TestDispatchRunbookLive:
    def test_set_assignment_called_for_each_directive(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "live.yaml",
            {
                "name": "live",
                "directives": [
                    {"target": "agent-1", "task": "task one"},
                    {"target": "agent-2", "task": "task two"},
                ],
            },
        )
        mock_result = {"ok": True}
        with patch(
            "aragora.swarm.session_coordinator.set_assignment",
            return_value=mock_result,
        ) as mock_sa:
            result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=False)

        assert mock_sa.call_count == 2
        assert result["dry_run"] is False
        assert result["directives"] == [mock_result, mock_result]

    def test_set_assignment_receives_correct_kwargs(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "kwargs.yaml",
            {
                "directives": [
                    {
                        "target": "worker-a",
                        "task": "write tests",
                        "scope": ["#42"],
                        "constraints": ["no refactor"],
                        "status": "active",
                    }
                ]
            },
        )
        mock_sa = MagicMock(return_value={})
        with patch("aragora.swarm.session_coordinator.set_assignment", mock_sa):
            dispatch_runbook(rb, issued_by="lead", repo_root=tmp_path, dry_run=False)

        mock_sa.assert_called_once_with(
            "worker-a",
            "write tests",
            scope=["#42"],
            constraints=["no refactor"],
            status="active",
            issued_by="lead",
            repo_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestDispatchRunbookValidation:
    def test_missing_target_raises(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "no_target.yaml",
            {"directives": [{"task": "do something"}]},
        )
        with pytest.raises(ValueError, match="target"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_missing_task_raises(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "no_task.yaml",
            {"directives": [{"target": "agent-1"}]},
        )
        with pytest.raises(ValueError, match="task"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_empty_target_raises(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "empty_target.yaml",
            {"directives": [{"target": "   ", "task": "do something"}]},
        )
        with pytest.raises(ValueError, match="target"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_empty_task_raises(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "empty_task.yaml",
            {"directives": [{"target": "agent-1", "task": ""}]},
        )
        with pytest.raises(ValueError, match="task"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_non_list_directives_raises(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "non_list.yaml",
            {"directives": {"target": "x", "task": "y"}},
        )
        with pytest.raises(ValueError, match="list"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_non_mapping_directive_item_raises(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "bad_item.yaml",
            {"directives": ["just a string"]},
        )
        with pytest.raises(ValueError, match="mapping"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_non_mapping_yaml_root_raises(self, tmp_path: Path):
        rb = tmp_path / "list_root.yaml"
        rb.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)

    def test_empty_directives_list_is_valid(self, tmp_path: Path):
        rb = _write_runbook(tmp_path / "empty.yaml", {"name": "empty", "directives": []})
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"] == []

    def test_null_directives_treated_as_empty(self, tmp_path: Path):
        rb = _write_runbook(tmp_path / "null_dirs.yaml", {"name": "n", "directives": None})
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"] == []


# ---------------------------------------------------------------------------
# _normalize_directive via dispatch_runbook (scope/constraints filtering)
# ---------------------------------------------------------------------------


class TestNormalizeDirectiveViaDispatch:
    def test_blank_scope_entries_filtered(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "scope.yaml",
            {"directives": [{"target": "a", "task": "t", "scope": ["#1", "  ", "#2"]}]},
        )
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["scope"] == ["#1", "#2"]

    def test_blank_constraints_entries_filtered(self, tmp_path: Path):
        rb = _write_runbook(
            tmp_path / "cons.yaml",
            {"directives": [{"target": "a", "task": "t", "constraints": ["c1", "", "c2"]}]},
        )
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["constraints"] == ["c1", "c2"]

    def test_missing_scope_defaults_to_empty_list(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["scope"] == []

    def test_missing_constraints_defaults_to_empty_list(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["constraints"] == []

    def test_status_defaults_to_active_when_no_default_status(self, tmp_path: Path):
        rb = _minimal_runbook(tmp_path)
        result = dispatch_runbook(rb, issued_by="ci", repo_root=tmp_path, dry_run=True)
        assert result["directives"][0]["status"] == "active"
