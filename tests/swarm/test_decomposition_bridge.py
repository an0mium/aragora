"""Tests for decomposition_bridge orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.nomic.task_decomposer import SubTask
from aragora.swarm.boss_validation import assess_issue_body_sanitation
from aragora.swarm.decomposition_bridge import (
    DecompositionBridge,
    DecompositionOutcome,
    _partition_paths,
)
from aragora.swarm.task_sanitizer import SanitizationOutcome, SanitizationResult


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    (tmp_path / "aragora" / "pkg").mkdir(parents=True)
    (tmp_path / "tests" / "pkg").mkdir(parents=True)
    (tmp_path / "aragora" / "pkg" / "module.py").write_text(
        '"""module docs"""\n\ndef run() -> None:\n    pass\n'
    )
    (tmp_path / "aragora" / "pkg" / "helper.py").write_text("def helper() -> int:\n    return 1\n")
    (tmp_path / "tests" / "pkg" / "test_module.py").write_text("def test_run():\n    assert True\n")
    return tmp_path


@pytest.fixture
def bridge(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> DecompositionBridge:
    async def fake_refiner(
        title: str, body: str, *, repo_path: Path | None = None, timeout_seconds: float = 45.0
    ):  # noqa: ARG001
        return {
            "refined_prompt": body,
            "files_to_change": [],
            "test_patterns": [],
            "constraints": [],
            "context_gathered": True,
        }

    class PassSanitizer:
        def sanitize(self, title: str, body: str) -> SanitizationResult:
            composed = f"{title}\n\n{body}"
            return SanitizationResult(
                outcome=SanitizationOutcome.ACCEPTED,
                original_text=composed,
                sanitized_text=composed,
                reason="accepted",
                confidence=0.99,
                checks_failed=[],
            )

    monkeypatch.setattr("aragora.swarm.decomposition_bridge.refine_worker_prompt", fake_refiner)
    result = DecompositionBridge(repo_root)
    result._task_sanitizer = PassSanitizer()
    return result


def _make_subtask(
    *,
    title: str = "Target module",
    description: str = "Update the module safely and keep the change tightly scoped.",
    file_scope: list[str] | None = None,
    estimated_complexity: str = "low",
    success_criteria: dict[str, str] | None = None,
) -> SubTask:
    return SubTask(
        id="st-1",
        title=title,
        description=description,
        file_scope=list(file_scope or []),
        estimated_complexity=estimated_complexity,
        success_criteria=dict(success_criteria or {}),
    )


class _FakeDecomposer:
    def __init__(self, subtasks: list[SubTask] | None = None) -> None:
        self.subtasks = list(subtasks or [])
        self.calls: list[dict[str, object]] = []

    def analyze(self, task_description: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"task_description": task_description, **kwargs})
        return SimpleNamespace(subtasks=list(self.subtasks))


class TestHelpers:
    def test_partition_paths_marks_missing_files_as_new(self, repo_root: Path) -> None:
        file_scope, new_files = _partition_paths(
            repo_root,
            ["aragora/pkg/module.py", "tests/pkg/test_new.py"],
        )
        assert file_scope == ["aragora/pkg/module.py"]
        assert new_files == ["tests/pkg/test_new.py"]

    def test_partition_paths_respects_declared_new(self, repo_root: Path) -> None:
        file_scope, new_files = _partition_paths(
            repo_root,
            ["aragora/pkg/module.py", "tests/pkg/test_module.py"],
            declared_new=["tests/pkg/test_module.py"],
        )
        assert file_scope == ["aragora/pkg/module.py"]
        assert new_files == ["tests/pkg/test_module.py"]

    def test_build_parent_spec_extracts_file_scope_and_validation(
        self, bridge: DecompositionBridge
    ) -> None:
        spec = bridge._build_parent_spec(
            "Add tests for module",
            "## Task\n\nCover the module.\n\n### File Scope\n- `aragora/pkg/module.py`\n- `tests/pkg/test_module.py` (create)\n\n### Validation\n- python3 -m pytest -q tests/pkg/test_module.py",
        )
        assert "aragora/pkg/module.py" in spec.file_scope_hints
        assert "tests/pkg/test_module.py" in spec.file_scope_hints
        assert "python3 -m pytest -q tests/pkg/test_module.py" in spec.acceptance_criteria

    def test_build_parent_spec_extracts_constraints(self, bridge: DecompositionBridge) -> None:
        spec = bridge._build_parent_spec(
            "Add tests",
            "## Task\n\nDo the thing.\n\n### Constraints\n- Do not modify other files",
        )
        assert any("Do not modify other files" in item for item in spec.constraints)

    def test_render_candidate_body_has_required_sections(self, bridge: DecompositionBridge) -> None:
        candidate = pytest.importorskip("aragora.swarm.issue_scanner").BossIssueCandidate(
            category="decomposed_issue",
            title="Write tests for module.py",
            description="Add focused tests.",
            file_scope=["aragora/pkg/module.py"],
            new_files=["tests/pkg/test_module.py"],
            validation_command="python3 -m pytest -q tests/pkg/test_module.py",
            acceptance_criteria=["Tests pass"],
        )
        body = bridge._render_candidate_body(candidate)
        assert "## Task" in body
        assert "### File Scope" in body
        assert "### Validation" in body
        assert "### Acceptance Criteria" in body


class TestCandidateConstruction:
    @pytest.mark.asyncio
    async def test_candidate_prefers_explicit_validation_command(
        self, bridge: DecompositionBridge
    ) -> None:
        candidate = await bridge._candidate_from_parts(
            title="Write tests",
            category="test_coverage",
            description="Use `python3 -m pytest -q tests/pkg/test_module.py` after changes.",
            scope_paths=["aragora/pkg/module.py", "tests/pkg/test_module.py"],
            acceptance_criteria=[],
            estimated_complexity="small",
        )
        assert candidate.validation_command == "python3 -m pytest -q tests/pkg/test_module.py"

    @pytest.mark.asyncio
    async def test_candidate_uses_refiner_test_pattern_fallback(
        self, bridge: DecompositionBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_refiner(
            title: str, body: str, *, repo_path: Path | None = None, timeout_seconds: float = 45.0
        ):  # noqa: ARG001
            return {
                "refined_prompt": body,
                "files_to_change": [],
                "test_patterns": ["tests/pkg/test_module.py"],
                "constraints": [],
                "context_gathered": True,
            }

        monkeypatch.setattr("aragora.swarm.decomposition_bridge.refine_worker_prompt", fake_refiner)
        candidate = await bridge._candidate_from_parts(
            title="Write tests",
            category="test_coverage",
            description="Add focused tests.",
            scope_paths=["aragora/pkg/module.py"],
            acceptance_criteria=[],
            estimated_complexity="small",
        )
        assert candidate.validation_command == "python3 -m pytest -q tests/pkg/test_module.py"

    @pytest.mark.asyncio
    async def test_candidate_uses_pytest_for_test_file_scope(
        self, bridge: DecompositionBridge
    ) -> None:
        candidate = await bridge._candidate_from_parts(
            title="Write tests",
            category="test_coverage",
            description="Add focused tests.",
            scope_paths=["tests/pkg/test_module.py"],
            acceptance_criteria=[],
            estimated_complexity="small",
        )
        assert candidate.validation_command == "python3 -m pytest -q tests/pkg/test_module.py"

    @pytest.mark.asyncio
    async def test_candidate_uses_ruff_fallback_for_source_files(
        self, bridge: DecompositionBridge
    ) -> None:
        candidate = await bridge._candidate_from_parts(
            title="Update module",
            category="decomposed_issue",
            description="Make a tiny change.",
            scope_paths=["aragora/pkg/module.py"],
            acceptance_criteria=[],
            estimated_complexity="small",
        )
        assert candidate.validation_command == "python3 -m ruff check aragora/pkg/module.py"

    @pytest.mark.asyncio
    async def test_candidate_complexity_is_bounded(self, bridge: DecompositionBridge) -> None:
        candidate = await bridge._candidate_from_parts(
            title="Update module",
            category="decomposed_issue",
            description="Make a tiny change.",
            scope_paths=["aragora/pkg/module.py"],
            acceptance_criteria=[],
            estimated_complexity="medium",
        )
        assert candidate.estimated_complexity in {"small", "medium"}


class TestGateCandidate:
    def test_gate_candidate_rejects_quarantined_child(self, bridge: DecompositionBridge) -> None:
        class RejectSanitizer:
            def sanitize(self, title: str, body: str) -> SanitizationResult:
                return SanitizationResult(
                    outcome=SanitizationOutcome.QUARANTINED,
                    original_text=body,
                    sanitized_text=body,
                    reason="bad",
                    confidence=0.99,
                    checks_failed=["scope_too_broad"],
                )

        bridge._task_sanitizer = RejectSanitizer()
        candidate = pytest.importorskip("aragora.swarm.issue_scanner").BossIssueCandidate(
            category="decomposed_issue",
            title="Bad child",
            description="Bad child description long enough to pass.",
            file_scope=["aragora/pkg/module.py"],
            validation_command="python3 -m ruff check aragora/pkg/module.py",
        )
        assert bridge._gate_candidate(candidate) is None

    def test_gate_candidate_rejects_dropped_child(self, bridge: DecompositionBridge) -> None:
        class DropSanitizer:
            def sanitize(self, title: str, body: str) -> SanitizationResult:
                return SanitizationResult(
                    outcome=SanitizationOutcome.DROPPED,
                    original_text=body,
                    sanitized_text=body,
                    reason="duplicate",
                    confidence=0.99,
                    checks_failed=["duplicate"],
                )

        bridge._task_sanitizer = DropSanitizer()
        candidate = pytest.importorskip("aragora.swarm.issue_scanner").BossIssueCandidate(
            category="decomposed_issue",
            title="Dropped child",
            description="Bad child description long enough to pass.",
            file_scope=["aragora/pkg/module.py"],
            validation_command="python3 -m ruff check aragora/pkg/module.py",
        )
        assert bridge._gate_candidate(candidate) is None

    def test_gate_candidate_adopts_rewritten_validation(self, bridge: DecompositionBridge) -> None:
        class RewriteSanitizer:
            def sanitize(self, title: str, body: str) -> SanitizationResult:
                rewritten = (
                    f"{title}\n\n"
                    "## Task\n\nRefined task body that is long enough for dispatch gating to accept it safely.\n\n"
                    "### File Scope\n- `aragora/pkg/module.py`\n\n"
                    "### Validation\n- python3 -m pytest -q tests/pkg/test_module.py"
                )
                return SanitizationResult(
                    outcome=SanitizationOutcome.REWRITTEN,
                    original_text=body,
                    sanitized_text=rewritten,
                    reason="added validation",
                    confidence=0.9,
                    checks_failed=["missing_validation"],
                )

        bridge._task_sanitizer = RewriteSanitizer()
        candidate = pytest.importorskip("aragora.swarm.issue_scanner").BossIssueCandidate(
            category="decomposed_issue",
            title="Rewritten child",
            description="Refined task body that is long enough for dispatch gating to accept it safely.",
            file_scope=["aragora/pkg/module.py"],
            validation_command="",
        )
        gated = bridge._gate_candidate(candidate)
        assert gated is not None
        assert gated.validation_command == "python3 -m pytest -q tests/pkg/test_module.py"

    def test_gate_candidate_adopts_rewritten_scope(self, bridge: DecompositionBridge) -> None:
        class RewriteScopeSanitizer:
            def sanitize(self, title: str, body: str) -> SanitizationResult:
                rewritten = (
                    f"{title}\n\n"
                    "## Task\n\nRefined task body that is long enough for dispatch gating to accept it safely.\n\n"
                    "### File Scope\n- `aragora/pkg/helper.py`\n\n"
                    "### Validation\n- python3 -m ruff check aragora/pkg/helper.py"
                )
                return SanitizationResult(
                    outcome=SanitizationOutcome.REWRITTEN,
                    original_text=body,
                    sanitized_text=rewritten,
                    reason="narrowed scope",
                    confidence=0.9,
                    checks_failed=["scope_too_broad"],
                )

        bridge._task_sanitizer = RewriteScopeSanitizer()
        candidate = pytest.importorskip("aragora.swarm.issue_scanner").BossIssueCandidate(
            category="decomposed_issue",
            title="Rewritten scope",
            description="Refined task body that is long enough for dispatch gating to accept it safely.",
            file_scope=["aragora/pkg/module.py", "aragora/pkg/helper.py"],
            validation_command="python3 -m ruff check aragora/pkg/module.py",
        )
        gated = bridge._gate_candidate(candidate)
        assert gated is not None
        assert gated.file_scope == ["aragora/pkg/helper.py"]

    def test_gate_candidate_requires_sanitation_ok(
        self, bridge: DecompositionBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "aragora.swarm.decomposition_bridge.assess_issue_body_sanitation",
            lambda body: (False, "task_truncated"),
        )
        candidate = pytest.importorskip("aragora.swarm.issue_scanner").BossIssueCandidate(
            category="decomposed_issue",
            title="Bad child",
            description="Refined task body that is long enough for dispatch gating to accept it safely.",
            file_scope=["aragora/pkg/module.py"],
            validation_command="python3 -m ruff check aragora/pkg/module.py",
        )
        assert bridge._gate_candidate(candidate) is None


class TestDecomposeIssue:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_subtasks_and_no_scope(
        self, bridge: DecompositionBridge
    ) -> None:
        bridge._task_decomposer = _FakeDecomposer([])
        result = await bridge.decompose_issue("Vague parent", "## Task\n\nThink about it.")
        assert result == []

    @pytest.mark.asyncio
    async def test_direct_child_from_subtask(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(
                    file_scope=["aragora/pkg/module.py"], success_criteria={"pytest": "passes"}
                )
            ]
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nImprove the module with a tightly scoped change that preserves existing behavior.\n\n### File Scope\n- `aragora/pkg/module.py`",
        )
        assert len(result) == 1
        assert result[0].file_scope == ["aragora/pkg/module.py"]
        assert result[0].validation_command

    @pytest.mark.asyncio
    async def test_micro_decomposer_used_for_broad_subtask(
        self, bridge: DecompositionBridge
    ) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(
                    title="Broad child",
                    file_scope=["aragora/pkg/module.py", "tests/pkg/test_module.py"],
                    estimated_complexity="high",
                )
            ]
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nImprove the module with a tightly scoped change that preserves existing behavior.\n\n### File Scope\n- `aragora/pkg/module.py`\n- `tests/pkg/test_module.py`",
        )
        assert len(result) >= 1
        assert all("validation" not in candidate.title.lower() for candidate in result)

    @pytest.mark.asyncio
    async def test_falls_back_to_micro_decomposer_without_subtasks(
        self, bridge: DecompositionBridge
    ) -> None:
        bridge._task_decomposer = _FakeDecomposer([])
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nImprove the module with a tightly scoped change that preserves existing behavior.\n\n### File Scope\n- `aragora/pkg/module.py`\n- `tests/pkg/test_module.py`",
        )
        assert len(result) >= 1
        assert any(
            "aragora/pkg/module.py" in candidate.file_scope
            or "tests/pkg/test_module.py" in candidate.file_scope
            for candidate in result
        )

    @pytest.mark.asyncio
    async def test_skips_validation_only_work_orders(
        self, bridge: DecompositionBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(
                    file_scope=["aragora/pkg/module.py", "tests/pkg/test_module.py"],
                    estimated_complexity="high",
                )
            ]
        )

        def fake_micro(
            *,
            goal: str,
            file_scope_hints: list[str],
            acceptance_criteria: list[str] | None = None,
            constraints: list[str] | None = None,
            repo_root: Path | None = None,
        ):  # noqa: ARG001
            return [
                {
                    "title": "Update module.py",
                    "description": "Refined task body that is long enough for dispatch gating to accept it safely.",
                    "file_scope": ["aragora/pkg/module.py"],
                    "estimated_complexity": "low",
                },
                {
                    "title": "Run validation and fix failures",
                    "description": "Validation lane that should be skipped because it is not an actionable child issue.",
                    "file_scope": ["aragora/pkg/module.py"],
                    "estimated_complexity": "low",
                },
            ]

        monkeypatch.setattr(
            "aragora.swarm.decomposition_bridge.build_micro_work_orders", fake_micro
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert len(result) == 1
        assert result[0].title == "Update module.py"

    @pytest.mark.asyncio
    async def test_max_children_cap_works(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(title="one", file_scope=["aragora/pkg/module.py"]),
                _make_subtask(title="two", file_scope=["aragora/pkg/helper.py"]),
                _make_subtask(title="three", file_scope=["tests/pkg/test_module.py"]),
            ]
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
            max_children=2,
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_file_scope_is_preserved(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [_make_subtask(file_scope=["aragora/pkg/helper.py"])]
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert result[0].file_scope == ["aragora/pkg/helper.py"]

    @pytest.mark.asyncio
    async def test_suppresses_overlapping_paths(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(title="first", file_scope=["aragora/pkg/module.py"]),
                _make_subtask(title="second", file_scope=["aragora/pkg/module.py"]),
            ]
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_rejects_children_without_validation_after_gate(
        self, bridge: DecompositionBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [_make_subtask(file_scope=["aragora/pkg/module.py"])]
        )
        monkeypatch.setattr(
            bridge,
            "_choose_validation_command",
            lambda **kwargs: "",
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_fingerprints_are_unique(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(title="first", file_scope=["aragora/pkg/module.py"]),
                _make_subtask(title="second", file_scope=["aragora/pkg/helper.py"]),
            ]
        )
        result = await bridge.decompose_issue(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert len(result) == 2
        assert result[0].fingerprint != result[1].fingerprint

    def test_sync_wrapper(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [_make_subtask(file_scope=["aragora/pkg/module.py"])]
        )
        result = bridge.decompose_issue_sync(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert len(result) == 1

    def test_sync_wrapper_with_stats_counts_rejections(self, bridge: DecompositionBridge) -> None:
        bridge._task_decomposer = _FakeDecomposer(
            [
                _make_subtask(title="first", file_scope=["aragora/pkg/module.py"]),
                _make_subtask(title="second", file_scope=["aragora/pkg/module.py"]),
            ]
        )
        result = bridge.decompose_issue_sync_with_stats(
            "Parent",
            "## Task\n\nParent task body that is long enough for dispatch gating to accept it safely.",
        )
        assert isinstance(result, DecompositionOutcome)
        assert len(result.children) == 1
        assert result.stats.raw_candidates == 2
        assert result.stats.accepted_candidates == 1
        assert result.stats.rejected_candidates == 1
        assert result.stats.overlap_rejections == 1


class TestRealIssueCorpus:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("title", "body"),
        [
            (
                "Add unit tests for server/fastapi/routes/dr.py",
                "## Task\n\n"
                "Add comprehensive unit tests for `aragora/server/fastapi/routes/dr.py`.\n\n"
                "### Requirements\n"
                "1. Read the module and identify all public functions.\n"
                "2. Create a test file with broad coverage.\n\n"
                "### File Scope\n"
                "- `aragora/pkg/module.py`\n"
                "- `tests/pkg/test_module.py` (create)\n",
            ),
            (
                "Add request body validation to handler-like endpoints",
                "## Task\n\n"
                "Add request body validation to `aragora/pkg/module.py` and supporting tests.\n\n"
                "### File Scope\n"
                "- `aragora/pkg/module.py`\n"
                "- `tests/pkg/test_module.py` (create)\n\n"
                "### Acceptance Criteria\n"
                "- invalid input is rejected clearly\n",
            ),
            (
                "Improve module and helper together",
                "## Task\n\n"
                "Improve the implementation while preserving existing behavior and keeping the work reviewable.\n\n"
                "### File Scope\n"
                "- `aragora/pkg/module.py`\n"
                "- `aragora/pkg/helper.py`\n"
                "- `tests/pkg/test_module.py` (create)\n",
            ),
            (
                "Replace silent exception swallowing in scheduler_bridge.py",
                "## Task\n\n"
                "Replace `except ...: pass` patterns with proper error handling in `aragora/pkg/helper.py`.\n\n"
                "### Requirements\n"
                "1. Read the file and find all silent exception swallowing.\n"
                "2. Add logging or specific handling where appropriate.\n\n"
                "### File Scope\n"
                "- `aragora/pkg/helper.py`\n\n"
                "### Validation\n"
                "- python3 -m ruff check aragora/pkg/helper.py\n",
            ),
        ],
    )
    async def test_realish_parent_templates_emit_bounded_children(
        self,
        bridge: DecompositionBridge,
        title: str,
        body: str,
    ) -> None:
        bridge._task_decomposer = _FakeDecomposer([])

        result = await bridge.decompose_issue(title, body, max_children=4)

        assert result
        for candidate in result:
            assert candidate.validation_command
            assert candidate.estimated_complexity in {"small", "medium"}
            assert 1 <= len(candidate.file_scope) + len(candidate.new_files) <= 5
            assert candidate.file_scope or candidate.new_files
            rendered = bridge._render_candidate_body(candidate)
            ok, reason = assess_issue_body_sanitation(rendered)
            assert ok, reason
