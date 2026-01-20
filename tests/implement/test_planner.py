"""
Tests for implementation planner.

Tests cover:
- extract_json function
- validate_plan function
- create_single_task_plan function
"""

import hashlib
import tempfile
from pathlib import Path

import pytest

from aragora.implement.planner import (
    create_single_task_plan,
    extract_json,
    validate_plan,
)
from aragora.implement.types import ImplementPlan


# ============================================================================
# extract_json Tests
# ============================================================================


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extracts_raw_json(self):
        """Test extracting raw JSON object."""
        text = '{"key": "value"}'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extracts_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = """Here's the plan:
```json
{"tasks": []}
```
Done!"""
        result = extract_json(text)
        assert result == '{"tasks": []}'

    def test_extracts_json_from_unmarked_code_block(self):
        """Test extracting JSON from code block without language marker."""
        text = """
```
{"id": "test"}
```
"""
        result = extract_json(text)
        assert result == '{"id": "test"}'

    def test_extracts_json_with_surrounding_text(self):
        """Test extracting JSON with surrounding prose."""
        text = """
Based on my analysis, here is the implementation plan:

{"tasks": [{"id": "task-1"}]}

Let me know if you need changes.
"""
        result = extract_json(text)
        assert '{"tasks"' in result

    def test_extracts_nested_json(self):
        """Test extracting nested JSON object."""
        text = '{"outer": {"inner": {"deep": 123}}}'
        result = extract_json(text)
        assert result == '{"outer": {"inner": {"deep": 123}}}'

    def test_returns_original_if_no_json(self):
        """Test returns original text if no JSON found."""
        text = "No JSON here at all"
        result = extract_json(text)
        assert result == "No JSON here at all"

    def test_prefers_code_block_over_raw(self):
        """Test code block JSON is preferred."""
        text = """
Inline {"inline": true}

```json
{"codeblock": true}
```
"""
        result = extract_json(text)
        assert '{"codeblock": true}' == result

    def test_handles_multiline_json(self):
        """Test extracting multiline JSON."""
        text = """```json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "First task"
    }
  ]
}
```"""
        result = extract_json(text)
        assert '"tasks"' in result
        assert '"id": "task-1"' in result


# ============================================================================
# validate_plan Tests
# ============================================================================


class TestValidatePlan:
    """Tests for validate_plan function."""

    def test_valid_plan(self):
        """Test valid plan returns no errors."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First task",
                    "files": ["a.py"],
                    "complexity": "simple",
                    "dependencies": [],
                },
                {
                    "id": "task-2",
                    "description": "Second task",
                    "files": ["b.py"],
                    "complexity": "moderate",
                    "dependencies": ["task-1"],
                },
            ]
        }
        errors = validate_plan(plan)
        assert errors == []

    def test_missing_tasks_key(self):
        """Test error for missing tasks key."""
        plan = {"other": "data"}
        errors = validate_plan(plan)
        assert "Missing 'tasks' key in plan" in errors

    def test_tasks_not_a_list(self):
        """Test error for tasks not being a list."""
        plan = {"tasks": "not a list"}
        errors = validate_plan(plan)
        assert "'tasks' must be a list" in errors

    def test_empty_tasks(self):
        """Test error for empty tasks list."""
        plan = {"tasks": []}
        errors = validate_plan(plan)
        assert "Plan has no tasks" in errors

    def test_task_not_dict(self):
        """Test error for task not being a dict."""
        plan = {"tasks": ["not a dict"]}
        errors = validate_plan(plan)
        assert "Task 0 is not a dict" in errors

    def test_task_missing_id(self):
        """Test error for task missing id."""
        plan = {
            "tasks": [
                {"description": "No id", "files": [], "complexity": "simple"}
            ]
        }
        errors = validate_plan(plan)
        assert "Task 0 missing 'id'" in errors

    def test_task_missing_description(self):
        """Test error for task missing description."""
        plan = {
            "tasks": [
                {"id": "t1", "files": [], "complexity": "simple"}
            ]
        }
        errors = validate_plan(plan)
        assert "Task 0 missing 'description'" in errors

    def test_task_missing_files(self):
        """Test error for task missing files."""
        plan = {
            "tasks": [
                {"id": "t1", "description": "Test", "complexity": "simple"}
            ]
        }
        errors = validate_plan(plan)
        assert "Task 0 missing or invalid 'files'" in errors

    def test_task_files_not_list(self):
        """Test error for files not being a list."""
        plan = {
            "tasks": [
                {"id": "t1", "description": "Test", "files": "file.py", "complexity": "simple"}
            ]
        }
        errors = validate_plan(plan)
        assert "Task 0 missing or invalid 'files'" in errors

    def test_task_invalid_complexity(self):
        """Test error for invalid complexity value."""
        plan = {
            "tasks": [
                {"id": "t1", "description": "Test", "files": [], "complexity": "invalid"}
            ]
        }
        errors = validate_plan(plan)
        assert "Task 0 has invalid complexity: invalid" in errors

    def test_duplicate_task_ids(self):
        """Test error for duplicate task IDs."""
        plan = {
            "tasks": [
                {"id": "t1", "description": "First", "files": [], "complexity": "simple"},
                {"id": "t1", "description": "Duplicate", "files": [], "complexity": "simple"},
            ]
        }
        errors = validate_plan(plan)
        assert "Duplicate task id: t1" in errors

    def test_valid_complexities(self):
        """Test all valid complexity values."""
        for complexity in ["simple", "moderate", "complex"]:
            plan = {
                "tasks": [
                    {"id": "t1", "description": "Test", "files": ["f.py"], "complexity": complexity}
                ]
            }
            errors = validate_plan(plan)
            assert errors == [], f"Complexity '{complexity}' should be valid"

    def test_multiple_errors(self):
        """Test multiple errors are collected."""
        plan = {
            "tasks": [
                {"id": "t1"},  # Missing description, files, complexity
                {"description": "No id", "files": [], "complexity": "bad"},  # Missing id, bad complexity
            ]
        }
        errors = validate_plan(plan)
        assert len(errors) >= 4


# ============================================================================
# create_single_task_plan Tests
# ============================================================================


class TestCreateSingleTaskPlan:
    """Tests for create_single_task_plan function."""

    def test_creates_plan_with_single_task(self):
        """Test creates a plan with exactly one task."""
        design = "Implement feature X"
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert isinstance(plan, ImplementPlan)
        assert len(plan.tasks) == 1

    def test_task_has_correct_id(self):
        """Test single task has expected ID."""
        design = "Implement feature X"
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert plan.tasks[0].id == "task-1"

    def test_task_has_generic_description(self):
        """Test single task has generic description."""
        design = "Implement feature X"
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert "implement" in plan.tasks[0].description.lower()

    def test_task_complexity_is_complex(self):
        """Test single task is marked as complex."""
        design = "Implement feature X"
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert plan.tasks[0].complexity == "complex"

    def test_task_has_empty_files(self):
        """Test single task has empty files list."""
        design = "Implement feature X"
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert plan.tasks[0].files == []

    def test_task_has_no_dependencies(self):
        """Test single task has no dependencies."""
        design = "Implement feature X"
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert plan.tasks[0].dependencies == []

    def test_design_hash_is_deterministic(self):
        """Test design hash is deterministic."""
        design = "Implement feature X"
        expected_hash = hashlib.sha256(design.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = create_single_task_plan(design, Path(tmpdir))

        assert plan.design_hash == expected_hash

    def test_different_designs_different_hashes(self):
        """Test different designs produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan1 = create_single_task_plan("Design A", Path(tmpdir))
            plan2 = create_single_task_plan("Design B", Path(tmpdir))

        assert plan1.design_hash != plan2.design_hash


# ============================================================================
# Integration Tests
# ============================================================================


class TestPlannerIntegration:
    """Integration tests for planner functionality."""

    def test_extract_and_validate_valid(self):
        """Test extract + validate with valid JSON."""
        response = """
Here's your plan:
```json
{
  "tasks": [
    {
      "id": "task-1",
      "description": "Create the module",
      "files": ["src/module.py"],
      "complexity": "simple",
      "dependencies": []
    }
  ]
}
```
"""
        import json

        json_str = extract_json(response)
        plan_data = json.loads(json_str)
        errors = validate_plan(plan_data)

        assert errors == []
        assert len(plan_data["tasks"]) == 1

    def test_extract_and_validate_invalid(self):
        """Test extract + validate with invalid JSON structure."""
        response = """
```json
{
  "tasks": [
    {
      "description": "Missing id and files"
    }
  ]
}
```
"""
        import json

        json_str = extract_json(response)
        plan_data = json.loads(json_str)
        errors = validate_plan(plan_data)

        assert len(errors) > 0
