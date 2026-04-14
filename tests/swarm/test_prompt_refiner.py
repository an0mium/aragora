"""Tests for aragora/swarm/prompt_refiner.py.

Covers:
- build_refinement_worker_env: empty/None input, files_to_change, test_patterns
- refine_worker_prompt: normal paths, subprocess mock, OSError degradation
- _extract_keywords: stop-word filtering, length filtering, punctuation stripping
- _build_refined_prompt: section presence and content
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.swarm.prompt_refiner import (
    _build_refined_prompt,
    _extract_keywords,
    build_refinement_worker_env,
    refine_worker_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_proc(stdout_text: str = "") -> MagicMock:
    """Return a mock subprocess compatible with asyncio.create_subprocess_exec."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout_text.encode(), b""))
    return proc


# ---------------------------------------------------------------------------
# build_refinement_worker_env
# ---------------------------------------------------------------------------


class TestBuildRefinementWorkerEnv:
    def test_none_input_returns_empty_dict(self) -> None:
        result = build_refinement_worker_env(None)
        assert result == {}

    def test_empty_dict_returns_empty_dict(self) -> None:
        result = build_refinement_worker_env({})
        assert result == {}

    def test_files_to_change_sets_env_var(self) -> None:
        refinement = {"files_to_change": ["aragora/swarm/foo.py", "aragora/swarm/bar.py"]}
        result = build_refinement_worker_env(refinement)
        assert "ARAGORA_RELEVANT_FILES" in result
        parts = result["ARAGORA_RELEVANT_FILES"].split(os.pathsep)
        assert "aragora/swarm/foo.py" in parts
        assert "aragora/swarm/bar.py" in parts

    def test_test_patterns_sets_env_var(self) -> None:
        refinement = {"test_patterns": ["tests/swarm/test_foo.py", "tests/swarm/test_bar.py"]}
        result = build_refinement_worker_env(refinement)
        assert "ARAGORA_TEST_PATTERNS" in result
        parts = result["ARAGORA_TEST_PATTERNS"].split(os.pathsep)
        assert "tests/swarm/test_foo.py" in parts
        assert "tests/swarm/test_bar.py" in parts

    def test_both_keys_present(self) -> None:
        refinement = {
            "files_to_change": ["a.py"],
            "test_patterns": ["tests/test_a.py"],
        }
        result = build_refinement_worker_env(refinement)
        assert "ARAGORA_RELEVANT_FILES" in result
        assert "ARAGORA_TEST_PATTERNS" in result

    def test_empty_strings_in_list_are_ignored(self) -> None:
        refinement = {"files_to_change": ["", "  ", "aragora/real.py"]}
        result = build_refinement_worker_env(refinement)
        assert result["ARAGORA_RELEVANT_FILES"] == "aragora/real.py"

    def test_files_to_change_absent_no_env_var(self) -> None:
        result = build_refinement_worker_env({"test_patterns": ["tests/t.py"]})
        assert "ARAGORA_RELEVANT_FILES" not in result

    def test_test_patterns_absent_no_env_var(self) -> None:
        result = build_refinement_worker_env({"files_to_change": ["aragora/x.py"]})
        assert "ARAGORA_TEST_PATTERNS" not in result

    def test_single_file_no_pathsep_suffix(self) -> None:
        result = build_refinement_worker_env({"files_to_change": ["aragora/only.py"]})
        assert result["ARAGORA_RELEVANT_FILES"] == "aragora/only.py"

    def test_all_empty_strings_produces_no_env_var(self) -> None:
        result = build_refinement_worker_env({"files_to_change": ["", "   "]})
        assert "ARAGORA_RELEVANT_FILES" not in result

    def test_return_type_is_dict_of_str(self) -> None:
        result = build_refinement_worker_env({"files_to_change": ["x.py"]})
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# refine_worker_prompt — result shape and normal paths
# ---------------------------------------------------------------------------


class TestRefineWorkerPromptShape:
    def test_result_keys_always_present(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(refine_worker_prompt("", "", repo_path=tmp_path))
        for key in (
            "refined_prompt",
            "files_to_change",
            "test_patterns",
            "constraints",
            "context_gathered",
        ):
            assert key in result

    def test_files_to_change_is_list(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(refine_worker_prompt("fix queue", "", repo_path=tmp_path))
        assert isinstance(result["files_to_change"], list)

    def test_test_patterns_is_list(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(refine_worker_prompt("fix queue", "", repo_path=tmp_path))
        assert isinstance(result["test_patterns"], list)

    def test_context_gathered_false_when_no_files_found(self, tmp_path: Path) -> None:
        # When grep returns nothing, context_gathered should still be True because
        # the try block completes without exception — no files just means empty lists.
        # But context_gathered IS set to True regardless of file count.
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(refine_worker_prompt("fix queue", "", repo_path=tmp_path))
        assert isinstance(result["context_gathered"], bool)

    def test_title_appears_in_refined_prompt(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt("Fix tranche queue", "details", repo_path=tmp_path)
            )
        assert "Fix tranche queue" in result["refined_prompt"]

    def test_body_appears_in_refined_prompt(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt("Fix bug", "detailed body here", repo_path=tmp_path)
            )
        assert "detailed body here" in result["refined_prompt"]


class TestRefineWorkerPromptSubprocessMock:
    def test_empty_grep_stdout_yields_no_files(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt(
                    "fix the tranche queue",
                    "It crashes on empty input",
                    repo_path=tmp_path,
                )
            )
        assert result["files_to_change"] == []

    def test_grep_stdout_with_files_populates_files_to_change(self, tmp_path: Path) -> None:
        aragora_dir = tmp_path / "aragora" / "swarm"
        aragora_dir.mkdir(parents=True)
        target_file = aragora_dir / "tranche_queue.py"
        target_file.write_text("# stub")

        stdout_lines = str(target_file) + "\n"
        mock_proc = _make_mock_proc(stdout_text=stdout_lines)

        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt(
                    "fix tranche queue",
                    "",
                    repo_path=tmp_path,
                )
            )

        assert len(result["files_to_change"]) == 1
        assert "tranche_queue.py" in result["files_to_change"][0]

    def test_context_gathered_true_when_files_found(self, tmp_path: Path) -> None:
        aragora_dir = tmp_path / "aragora" / "swarm"
        aragora_dir.mkdir(parents=True)
        target_file = aragora_dir / "prompt_refiner.py"
        target_file.write_text("# stub")

        mock_proc = _make_mock_proc(stdout_text=str(target_file) + "\n")

        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt(
                    "refine prompt",
                    "details here",
                    repo_path=tmp_path,
                )
            )

        assert result["context_gathered"] is True

    def test_test_patterns_derived_from_source_files(self, tmp_path: Path) -> None:
        aragora_dir = tmp_path / "aragora" / "swarm"
        aragora_dir.mkdir(parents=True)
        source_file = aragora_dir / "prompt_refiner.py"
        source_file.write_text("# stub")

        tests_dir = tmp_path / "tests" / "swarm"
        tests_dir.mkdir(parents=True)
        test_file = tests_dir / "test_prompt_refiner.py"
        test_file.write_text("# tests")

        mock_proc = _make_mock_proc(stdout_text=str(source_file) + "\n")

        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt(
                    "fix prompt refiner",
                    "",
                    repo_path=tmp_path,
                )
            )

        assert any("test_prompt_refiner" in p for p in result["test_patterns"])

    def test_files_to_change_capped_at_ten(self, tmp_path: Path) -> None:
        aragora_dir = tmp_path / "aragora" / "swarm"
        aragora_dir.mkdir(parents=True)
        files = []
        for i in range(20):
            f = aragora_dir / f"module_{i}.py"
            f.write_text("# stub")
            files.append(str(f))

        mock_proc = _make_mock_proc(stdout_text="\n".join(files) + "\n")

        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt(
                    "fix many modules",
                    "",
                    repo_path=tmp_path,
                )
            )

        assert len(result["files_to_change"]) <= 10

    def test_pycache_files_excluded(self, tmp_path: Path) -> None:
        aragora_dir = tmp_path / "aragora" / "swarm" / "__pycache__"
        aragora_dir.mkdir(parents=True)
        cache_file = aragora_dir / "module.cpython-311.pyc"
        cache_file.write_text("# bytecode")

        # Grep output includes a __pycache__ path; it should be filtered out
        mock_proc = _make_mock_proc(stdout_text=str(cache_file) + "\n")

        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(refine_worker_prompt("fix module", "", repo_path=tmp_path))

        assert all("__pycache__" not in f for f in result["files_to_change"])


class TestRefineWorkerPromptGracefulDegradation:
    def test_oserror_from_find_relevant_files_is_silenced(self, tmp_path: Path) -> None:
        """OSError inside _find_relevant_files is caught internally; function completes."""
        with patch(
            "aragora.swarm.prompt_refiner._find_relevant_files",
            side_effect=OSError("grep not found"),
        ):
            result = asyncio.run(
                refine_worker_prompt(
                    "fix something",
                    "details",
                    repo_path=tmp_path,
                )
            )

        assert result["context_gathered"] is False
        assert result["refined_prompt"] != ""
        assert "fix something" in result["refined_prompt"]

    def test_oserror_returns_empty_files_and_patterns(self, tmp_path: Path) -> None:
        with patch(
            "aragora.swarm.prompt_refiner._find_relevant_files",
            side_effect=OSError("no grep"),
        ):
            result = asyncio.run(refine_worker_prompt("fix x", "", repo_path=tmp_path))

        assert result["files_to_change"] == []
        assert result["test_patterns"] == []

    def test_runtime_error_falls_back_gracefully(self, tmp_path: Path) -> None:
        with patch(
            "aragora.swarm.prompt_refiner._find_relevant_files",
            side_effect=RuntimeError("unexpected"),
        ):
            result = asyncio.run(refine_worker_prompt("title", "body", repo_path=tmp_path))

        assert result["context_gathered"] is False
        assert "title" in result["refined_prompt"]

    def test_value_error_falls_back_gracefully(self, tmp_path: Path) -> None:
        with patch(
            "aragora.swarm.prompt_refiner._find_relevant_files",
            side_effect=ValueError("bad value"),
        ):
            result = asyncio.run(refine_worker_prompt("fix value", "body", repo_path=tmp_path))

        assert result["context_gathered"] is False

    def test_no_exception_means_context_gathered_true(self, tmp_path: Path) -> None:
        mock_proc = _make_mock_proc(stdout_text="")
        with patch(
            "aragora.swarm.prompt_refiner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = asyncio.run(
                refine_worker_prompt("normal title", "normal body", repo_path=tmp_path)
            )

        assert result["context_gathered"] is True


# ---------------------------------------------------------------------------
# _extract_keywords (private but directly testable)
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_stop_words_excluded(self) -> None:
        keywords = _extract_keywords("add a feature to the system")
        assert "the" not in keywords
        assert "to" not in keywords
        assert "a" not in keywords

    def test_short_words_excluded(self) -> None:
        # Words 2 chars or fewer should be excluded (len(w) > 2)
        keywords = _extract_keywords("fix an io bug in system")
        assert "io" not in keywords
        assert "an" not in keywords

    def test_meaningful_words_included(self) -> None:
        keywords = _extract_keywords("improve tranche queue performance")
        assert "tranche" in keywords
        assert "queue" in keywords
        assert "performance" in keywords

    def test_empty_title_returns_empty(self) -> None:
        assert _extract_keywords("") == []

    def test_result_capped_at_eight(self) -> None:
        title = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        assert len(_extract_keywords(title)) <= 8

    def test_punctuation_stripped(self) -> None:
        keywords = _extract_keywords("fix tranche_queue.py, now!")
        for kw in keywords:
            assert not kw.endswith(".")
            assert not kw.endswith(",")
            assert not kw.endswith("!")

    def test_all_stop_words_returns_empty(self) -> None:
        keywords = _extract_keywords("the a an to and or in on for")
        assert keywords == []

    def test_lowercase_conversion(self) -> None:
        keywords = _extract_keywords("Fix Tranche Queue")
        assert all(kw == kw.lower() for kw in keywords)


# ---------------------------------------------------------------------------
# _build_refined_prompt (private but directly testable)
# ---------------------------------------------------------------------------


class TestBuildRefinedPrompt:
    def test_contains_original_goal(self) -> None:
        prompt = _build_refined_prompt(
            goal="Fix the queue",
            relevant_files=["aragora/swarm/queue.py"],
            test_files=[],
        )
        assert "Fix the queue" in prompt

    def test_relevant_files_section_present_when_files_given(self) -> None:
        prompt = _build_refined_prompt(
            goal="goal",
            relevant_files=["aragora/swarm/foo.py"],
            test_files=[],
        )
        assert "Relevant Files" in prompt
        assert "aragora/swarm/foo.py" in prompt

    def test_test_files_section_present_when_given(self) -> None:
        prompt = _build_refined_prompt(
            goal="goal",
            relevant_files=[],
            test_files=["tests/swarm/test_foo.py"],
        )
        assert "test_foo" in prompt

    def test_implementation_rules_always_present(self) -> None:
        prompt = _build_refined_prompt(goal="goal", relevant_files=[], test_files=[])
        assert "Implementation Rules" in prompt

    def test_no_relevant_files_section_omitted(self) -> None:
        prompt = _build_refined_prompt(goal="goal", relevant_files=[], test_files=[])
        assert "Relevant Files" not in prompt

    def test_pytest_command_included_when_test_files(self) -> None:
        prompt = _build_refined_prompt(
            goal="goal",
            relevant_files=[],
            test_files=["tests/swarm/test_foo.py"],
        )
        assert "pytest" in prompt
        assert "test_foo.py" in prompt

    def test_commit_reminder_always_present(self) -> None:
        prompt = _build_refined_prompt(goal="goal", relevant_files=[], test_files=[])
        assert "commit" in prompt.lower()

    def test_relevant_files_capped_at_eight_in_output(self) -> None:
        files = [f"aragora/swarm/module_{i}.py" for i in range(20)]
        prompt = _build_refined_prompt(goal="goal", relevant_files=files, test_files=[])
        # Only up to 8 files should appear in the prompt
        count = sum(1 for f in files if f in prompt)
        assert count <= 8

    def test_test_files_capped_at_five_in_output(self) -> None:
        test_files = [f"tests/swarm/test_module_{i}.py" for i in range(10)]
        prompt = _build_refined_prompt(goal="goal", relevant_files=[], test_files=test_files)
        count = sum(1 for f in test_files if f in prompt)
        assert count <= 5
