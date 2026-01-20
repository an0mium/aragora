"""Tests for implementation planner module."""

import hashlib
import pytest
from pathlib import Path

from aragora.implement.planner import (
    extract_json,
    validate_plan,
    create_single_task_plan,
    PLAN_PROMPT_TEMPLATE,
)
from aragora.implement.types import ImplementPlan, ImplementTask


class TestExtractJson:
    """Tests for JSON extraction from text."""

    def test_extracts_raw_json(self):
        """Should extract plain JSON object."""
        text = '{"tasks": []}'
        result = extract_json(text)
        assert result == '{"tasks": []}'

    def test_extracts_json_from_code_block(self):
        """Should extract JSON from markdown code block."""
        text = '''Here's the plan:
```json
{"tasks": [{"id": "task-1"}]}
```
Let me explain...'''
        result = extract_json(text)
        assert '{"tasks": [{"id": "task-1"}]}' in result

    def test_extracts_json_from_untagged_code_block(self):
        """Should extract JSON from untagged code block."""
        text = '''Output:
```
{"tasks": []}
```'''
        result = extract_json(text)
        assert '{"tasks": []}' in result

    def test_extracts_json_with_surrounding_text(self):
        """Should extract JSON when surrounded by text."""
        text = 'Prefix text {"key": "value"} suffix text'
        result = extract_json(text)
        assert '{"key": "value"}' in result

    def test_handles_nested_json(self):
        """Should extract nested JSON objects."""
        text = '{"outer": {"inner": "value"}}'
        result = extract_json(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_returns_original_if_no_json(self):
        """Should return original text if no JSON found."""
        text = "No JSON here"
        result = extract_json(text)
        assert result == text


class TestValidatePlan:
    """Tests for plan validation."""

    def test_valid_plan(self):
        """Valid plan should return no errors."""
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
        """Should error on missing tasks key."""
        plan = {"other": "data"}
        errors = validate_plan(plan)
        assert any("Missing 'tasks' key" in e for e in errors)

    def test_tasks_not_list(self):
        """Should error if tasks is not a list."""
        plan = {"tasks": "not a list"}
        errors = validate_plan(plan)
        assert any("must be a list" in e for e in errors)

    def test_empty_tasks(self):
        """Should error on empty tasks list."""
        plan = {"tasks": []}
        errors = validate_plan(plan)
        assert any("no tasks" in e for e in errors)

    def test_task_not_dict(self):
        """Should error if task is not a dict."""
        plan = {"tasks": ["not a dict"]}
        errors = validate_plan(plan)
        assert any("not a dict" in e for e in errors)

    def test_missing_task_id(self):
        """Should error on missing task id."""
        plan = {
            "tasks": [
                {
                    "description": "Task without id",
                    "files": [],
                    "complexity": "simple",
                }
            ]
        }
        errors = validate_plan(plan)
        assert any("missing 'id'" in e for e in errors)

    def test_missing_description(self):
        """Should error on missing description."""
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

    def test_missing_files(self):
        """Should error on missing files."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test",
                    "complexity": "simple",
                }
            ]
        }
        errors = validate_plan(plan)
        assert any("missing or invalid 'files'" in e for e in errors)

    def test_files_not_list(self):
        """Should error if files is not a list."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test",
                    "files": "not a list",
                    "complexity": "simple",
                }
            ]
        }
        errors = validate_plan(plan)
        assert any("missing or invalid 'files'" in e for e in errors)

    def test_invalid_complexity(self):
        """Should error on invalid complexity."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test",
                    "files": [],
                    "complexity": "invalid",
                }
            ]
        }
        errors = validate_plan(plan)
        assert any("invalid complexity" in e for e in errors)

    def test_duplicate_task_ids(self):
        """Should error on duplicate task IDs."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First",
                    "files": [],
                    "complexity": "simple",
                },
                {
                    "id": "task-1",
                    "description": "Duplicate",
                    "files": [],
                    "complexity": "simple",
                },
            ]
        }
        errors = validate_plan(plan)
        assert any("Duplicate task id" in e for e in errors)

    def test_valid_complexity_values(self):
        """All valid complexity values should pass."""
        for complexity in ("simple", "moderate", "complex"):
            plan = {
                "tasks": [
                    {
                        "id": f"task-{complexity}",
                        "description": f"Test {complexity}",
                        "files": ["file.py"],
                        "complexity": complexity,
                    }
                ]
            }
            errors = validate_plan(plan)
            assert errors == [], f"Failed for complexity: {complexity}"

    def test_multiple_valid_tasks(self):
        """Should validate multiple tasks correctly."""
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
                    "files": ["b.py", "c.py"],
                    "complexity": "moderate",
                    "dependencies": ["task-1"],
                },
            ]
        }
        errors = validate_plan(plan)
        assert errors == []


class TestCreateSingleTaskPlan:
    """Tests for fallback single-task plan creation."""

    def test_creates_plan_with_hash(self, tmp_path):
        """Should create plan with correct design hash."""
        design = "Test design content"
        expected_hash = hashlib.sha256(design.encode()).hexdigest()

        plan = create_single_task_plan(design, tmp_path)

        assert plan.design_hash == expected_hash

    def test_creates_single_task(self, tmp_path):
        """Should create exactly one task."""
        plan = create_single_task_plan("Design", tmp_path)
        assert len(plan.tasks) == 1

    def test_task_has_standard_id(self, tmp_path):
        """Task should have id 'task-1'."""
        plan = create_single_task_plan("Design", tmp_path)
        assert plan.tasks[0].id == "task-1"

    def test_task_is_complex(self, tmp_path):
        """Task should be marked as complex."""
        plan = create_single_task_plan("Design", tmp_path)
        assert plan.tasks[0].complexity == "complex"

    def test_task_has_empty_files(self, tmp_path):
        """Task should have empty files list."""
        plan = create_single_task_plan("Design", tmp_path)
        assert plan.tasks[0].files == []

    def test_task_has_no_dependencies(self, tmp_path):
        """Task should have no dependencies."""
        plan = create_single_task_plan("Design", tmp_path)
        assert plan.tasks[0].dependencies == []

    def test_task_has_generic_description(self, tmp_path):
        """Task should have generic description."""
        plan = create_single_task_plan("Design", tmp_path)
        assert "complete design" in plan.tasks[0].description.lower()

    def test_returns_implement_plan_instance(self, tmp_path):
        """Should return ImplementPlan instance."""
        plan = create_single_task_plan("Design", tmp_path)
        assert isinstance(plan, ImplementPlan)

    def test_different_designs_different_hashes(self, tmp_path):
        """Different designs should produce different hashes."""
        plan1 = create_single_task_plan("Design A", tmp_path)
        plan2 = create_single_task_plan("Design B", tmp_path)
        assert plan1.design_hash != plan2.design_hash


class TestPlanPromptTemplate:
    """Tests for the plan prompt template."""

    def test_template_has_design_placeholder(self):
        """Template should have design placeholder."""
        assert "{design}" in PLAN_PROMPT_TEMPLATE

    def test_template_has_repo_path_placeholder(self):
        """Template should have repo_path placeholder."""
        assert "{repo_path}" in PLAN_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        """Template should format without errors."""
        result = PLAN_PROMPT_TEMPLATE.format(
            design="Test design",
            repo_path="/test/path",
        )
        assert "Test design" in result
        assert "/test/path" in result

    def test_template_mentions_complexity_levels(self):
        """Template should document complexity levels."""
        assert "simple" in PLAN_PROMPT_TEMPLATE
        assert "moderate" in PLAN_PROMPT_TEMPLATE
        assert "complex" in PLAN_PROMPT_TEMPLATE

    def test_template_requests_json_output(self):
        """Template should request JSON output."""
        assert "JSON" in PLAN_PROMPT_TEMPLATE


class TestValidatePlanEdgeCases:
    """Edge case tests for plan validation."""

    def test_validates_with_dependencies(self):
        """Should validate tasks with dependencies."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Base",
                    "files": ["a.py"],
                    "complexity": "simple",
                    "dependencies": [],
                },
                {
                    "id": "task-2",
                    "description": "Depends on task-1",
                    "files": ["b.py"],
                    "complexity": "simple",
                    "dependencies": ["task-1"],
                },
            ]
        }
        errors = validate_plan(plan)
        assert errors == []

    def test_many_files_allowed(self):
        """Should allow tasks with many files."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Many files",
                    "files": [f"file{i}.py" for i in range(10)],
                    "complexity": "complex",
                }
            ]
        }
        errors = validate_plan(plan)
        assert errors == []

    def test_empty_description_fails(self):
        """Empty description string should still pass (not missing)."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "",
                    "files": [],
                    "complexity": "simple",
                }
            ]
        }
        errors = validate_plan(plan)
        # Empty string is not missing, so should pass
        assert not any("missing 'description'" in e for e in errors)
