"""
Tests for implement planner module.

Tests cover:
- extract_json function
- validate_plan function
- create_single_task_plan function
- Plan validation edge cases
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from aragora.implement.planner import (
    extract_json,
    validate_plan,
    create_single_task_plan,
)
from aragora.implement.types import ImplementTask


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extracts_raw_json(self):
        """Extracts JSON from raw response."""
        text = '{"tasks": []}'
        result = extract_json(text)
        assert result == '{"tasks": []}'

    def test_extracts_json_from_code_block(self):
        """Extracts JSON from markdown code block."""
        text = """Here is the plan:
```json
{"tasks": [{"id": "task-1"}]}
```
"""
        result = extract_json(text)
        assert '"tasks"' in result
        assert '"id"' in result

    def test_extracts_json_from_plain_code_block(self):
        """Extracts JSON from plain code block."""
        text = """```
{"tasks": []}
```"""
        result = extract_json(text)
        assert '"tasks"' in result

    def test_extracts_json_with_surrounding_text(self):
        """Extracts JSON object from text with surrounding content."""
        text = 'Some preamble {"tasks": [{"id": "t1"}]} and more text'
        result = extract_json(text)
        assert '"tasks"' in result

    def test_handles_no_json(self):
        """Returns original text when no JSON found."""
        text = "No JSON here"
        result = extract_json(text)
        assert result == text

    def test_handles_nested_json(self):
        """Handles nested JSON objects."""
        text = '{"tasks": [{"id": "t1", "files": ["a.py"]}]}'
        result = extract_json(text)
        assert result == text

    def test_prefers_code_block_over_raw(self):
        """Prefers JSON in code blocks over raw JSON."""
        text = """{"old": "json"}
```json
{"new": "json"}
```"""
        result = extract_json(text)
        assert '"new"' in result


class TestValidatePlan:
    """Tests for validate_plan function."""

    def test_valid_plan(self):
        """Valid plan returns no errors."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First task",
                    "files": ["a.py"],
                    "complexity": "simple",
                    "dependencies": [],
                }
            ]
        }

        errors = validate_plan(plan)
        assert errors == []

    def test_missing_tasks_key(self):
        """Missing tasks key returns error."""
        plan = {"other": "data"}

        errors = validate_plan(plan)
        assert len(errors) == 1
        assert "Missing 'tasks' key" in errors[0]

    def test_tasks_not_list(self):
        """Non-list tasks returns error."""
        plan = {"tasks": "not a list"}

        errors = validate_plan(plan)
        assert len(errors) == 1
        assert "'tasks' must be a list" in errors[0]

    def test_empty_tasks(self):
        """Empty tasks list returns error."""
        plan = {"tasks": []}

        errors = validate_plan(plan)
        assert len(errors) == 1
        assert "has no tasks" in errors[0]

    def test_task_not_dict(self):
        """Non-dict task returns error."""
        plan = {"tasks": ["not a dict"]}

        errors = validate_plan(plan)
        assert any("not a dict" in e for e in errors)

    def test_missing_task_id(self):
        """Missing task ID returns error."""
        plan = {
            "tasks": [
                {
                    "description": "No ID",
                    "files": [],
                    "complexity": "simple",
                }
            ]
        }

        errors = validate_plan(plan)
        assert any("missing 'id'" in e for e in errors)

    def test_missing_task_description(self):
        """Missing description returns error."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "files": [],
                    "complexity": "simple",
                }
            ]
        }

        errors = validate_plan(plan)
        assert any("missing 'description'" in e for e in errors)

    def test_missing_task_files(self):
        """Missing files returns error."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task",
                    "complexity": "simple",
                }
            ]
        }

        errors = validate_plan(plan)
        assert any("missing or invalid 'files'" in e for e in errors)

    def test_invalid_complexity(self):
        """Invalid complexity returns error."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task",
                    "files": ["a.py"],
                    "complexity": "invalid",
                }
            ]
        }

        errors = validate_plan(plan)
        assert any("invalid complexity" in e for e in errors)

    def test_duplicate_task_ids(self):
        """Duplicate task IDs return error."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First",
                    "files": ["a.py"],
                    "complexity": "simple",
                },
                {
                    "id": "task-1",
                    "description": "Duplicate",
                    "files": ["b.py"],
                    "complexity": "simple",
                },
            ]
        }

        errors = validate_plan(plan)
        assert any("Duplicate task id" in e for e in errors)

    def test_valid_complexities(self):
        """All valid complexities pass."""
        for complexity in ["simple", "moderate", "complex"]:
            plan = {
                "tasks": [
                    {
                        "id": "task-1",
                        "description": "Task",
                        "files": ["a.py"],
                        "complexity": complexity,
                    }
                ]
            }
            errors = validate_plan(plan)
            assert errors == []

    def test_multiple_errors(self):
        """Returns multiple errors for multiple issues."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    # Missing description
                    # Missing files
                    "complexity": "invalid",
                }
            ]
        }

        errors = validate_plan(plan)
        assert len(errors) >= 2


class TestCreateSingleTaskPlan:
    """Tests for create_single_task_plan function."""

    def test_creates_fallback_plan(self):
        """Creates single-task fallback plan."""
        design = "Implement a new feature"
        repo_path = Path("/repo")

        plan = create_single_task_plan(design, repo_path)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "task-1"
        assert plan.tasks[0].complexity == "complex"
        assert plan.tasks[0].files == []
        assert plan.tasks[0].dependencies == []

    def test_design_hash_is_deterministic(self):
        """Same design produces same hash."""
        design = "Test design"
        repo_path = Path("/repo")

        plan1 = create_single_task_plan(design, repo_path)
        plan2 = create_single_task_plan(design, repo_path)

        assert plan1.design_hash == plan2.design_hash

    def test_different_designs_different_hash(self):
        """Different designs produce different hashes."""
        repo_path = Path("/repo")

        plan1 = create_single_task_plan("Design A", repo_path)
        plan2 = create_single_task_plan("Design B", repo_path)

        assert plan1.design_hash != plan2.design_hash

    def test_task_description_is_generic(self):
        """Task description is generic."""
        plan = create_single_task_plan("Any design", Path("/repo"))

        assert "complete design" in plan.tasks[0].description.lower()


class TestPlanValidationEdgeCases:
    """Edge case tests for plan validation."""

    def test_files_wrong_type(self):
        """Files as string instead of list."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task",
                    "files": "not_a_list.py",
                    "complexity": "simple",
                }
            ]
        }

        errors = validate_plan(plan)
        assert any("invalid 'files'" in e for e in errors)

    def test_empty_task_id(self):
        """Empty task ID is still valid (has the key)."""
        plan = {
            "tasks": [
                {
                    "id": "",
                    "description": "Task",
                    "files": ["a.py"],
                    "complexity": "simple",
                }
            ]
        }

        errors = validate_plan(plan)
        # Empty ID is technically valid (key exists)
        assert not any("missing 'id'" in e for e in errors)

    def test_missing_complexity_defaults(self):
        """Missing complexity is flagged."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task",
                    "files": ["a.py"],
                    # No complexity
                }
            ]
        }

        errors = validate_plan(plan)
        assert any("invalid complexity" in e for e in errors)

    def test_empty_files_list_is_valid(self):
        """Empty files list is valid."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task",
                    "files": [],
                    "complexity": "simple",
                }
            ]
        }

        errors = validate_plan(plan)
        assert errors == []

    def test_many_tasks_validated(self):
        """Validates all tasks in a large plan."""
        plan = {
            "tasks": [
                {
                    "id": f"task-{i}",
                    "description": f"Task {i}",
                    "files": [f"file{i}.py"],
                    "complexity": "simple",
                    "dependencies": [f"task-{i - 1}"] if i > 0 else [],
                }
                for i in range(10)
            ]
        }

        errors = validate_plan(plan)
        assert errors == []
