"""Tests for aragora.nomic.testfixer.proposer module."""

from __future__ import annotations

import pytest

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aragora.nomic.testfixer.proposer import (
    FilePatch,
    PatchProposal,
    PatchStatus,
    ProposalDebate,
    SimpleCodeGenerator,
    PatchProposer,
)
from aragora.nomic.testfixer.analyzer import (
    FailureAnalysis,
    FailureCategory,
    FixTarget,
)
from aragora.nomic.testfixer.runner import TestFailure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure(
    test_file: str = "tests/test_example.py",
    test_name: str = "test_something",
) -> TestFailure:
    return TestFailure(
        test_name=test_name,
        test_file=test_file,
        error_type="AssertionError",
        error_message="assert 1 == 2",
        stack_trace="Traceback ...",
    )


def _make_analysis(
    category: FailureCategory = FailureCategory.TEST_ASSERTION,
    fix_target: FixTarget = FixTarget.TEST_FILE,
    test_file: str = "tests/test_example.py",
    root_cause_file: str = "src/example.py",
    confidence: float = 0.7,
) -> FailureAnalysis:
    return FailureAnalysis(
        failure=_make_failure(test_file=test_file),
        category=category,
        fix_target=fix_target,
        confidence=confidence,
        root_cause="Something went wrong",
        root_cause_file=root_cause_file,
        relevant_code={},
        suggested_approach="Fix the thing",
    )


# ===========================================================================
# FilePatch tests
# ===========================================================================


class TestFilePatch:
    """Tests for the FilePatch dataclass."""

    def test_diff_computation_lines_added(self):
        patch = FilePatch(
            file_path="foo.py",
            original_content="line1\n",
            patched_content="line1\nline2\n",
        )
        assert patch.lines_added == 1
        assert patch.lines_removed == 0
        assert len(patch.diff_lines) > 0

    def test_diff_computation_lines_removed(self):
        patch = FilePatch(
            file_path="foo.py",
            original_content="line1\nline2\n",
            patched_content="line1\n",
        )
        assert patch.lines_added == 0
        assert patch.lines_removed == 1

    def test_diff_computation_mixed_changes(self):
        patch = FilePatch(
            file_path="foo.py",
            original_content="alpha\nbeta\n",
            patched_content="alpha\ngamma\ndelta\n",
        )
        # beta removed, gamma and delta added
        assert patch.lines_removed == 1
        assert patch.lines_added == 2

    def test_diff_no_change(self):
        patch = FilePatch(
            file_path="foo.py",
            original_content="same\n",
            patched_content="same\n",
        )
        assert patch.lines_added == 0
        assert patch.lines_removed == 0
        assert patch.diff_lines == []

    def test_as_unified_diff(self):
        patch = FilePatch(
            file_path="foo.py",
            original_content="old\n",
            patched_content="new\n",
        )
        diff_str = patch.as_unified_diff()
        assert "a/foo.py" in diff_str
        assert "b/foo.py" in diff_str
        assert "-old" in diff_str
        assert "+new" in diff_str

    def test_apply_writes_file(self, tmp_path: Path):
        target = tmp_path / "src" / "hello.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original")

        patch = FilePatch(
            file_path="src/hello.py",
            original_content="original",
            patched_content="patched",
        )
        result = patch.apply(tmp_path)

        assert result is True
        assert target.read_text() == "patched"

    def test_apply_creates_parent_dirs(self, tmp_path: Path):
        patch = FilePatch(
            file_path="deep/nested/dir/file.py",
            original_content="",
            patched_content="content",
        )
        result = patch.apply(tmp_path)

        assert result is True
        assert (tmp_path / "deep" / "nested" / "dir" / "file.py").read_text() == "content"

    def test_revert_restores_original(self, tmp_path: Path):
        target = tmp_path / "file.py"
        target.write_text("patched")

        patch = FilePatch(
            file_path="file.py",
            original_content="original",
            patched_content="patched",
        )
        result = patch.revert(tmp_path)

        assert result is True
        assert target.read_text() == "original"

    def test_apply_returns_false_on_error(self, tmp_path: Path):
        # Use a path that cannot be written to (file as parent directory)
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file")

        patch = FilePatch(
            file_path="blocker/child/file.py",
            original_content="",
            patched_content="content",
        )
        result = patch.apply(tmp_path)
        assert result is False

    def test_revert_returns_false_on_error(self):
        # Revert on a non-existent base path should fail
        patch = FilePatch(
            file_path="file.py",
            original_content="original",
            patched_content="patched",
        )
        result = patch.revert(Path("/nonexistent_path_for_testing_12345"))
        assert result is False

    def test_pre_supplied_diff_lines_not_recomputed(self):
        """If diff_lines are already set, __post_init__ should not overwrite."""
        preset = ["preset diff line"]
        patch = FilePatch(
            file_path="foo.py",
            original_content="a\n",
            patched_content="b\n",
            diff_lines=preset,
        )
        assert patch.diff_lines == preset


# ===========================================================================
# PatchProposal tests
# ===========================================================================


class TestPatchProposal:
    """Tests for the PatchProposal dataclass."""

    def _make_proposal(self, patches: list[FilePatch] | None = None) -> PatchProposal:
        return PatchProposal(
            id="test_1",
            analysis=_make_analysis(),
            patches=patches or [],
        )

    def test_total_changes_empty(self):
        proposal = self._make_proposal([])
        assert proposal.total_changes() == (0, 0)

    def test_total_changes_sums_patches(self):
        p1 = FilePatch(
            file_path="a.py",
            original_content="line1\n",
            patched_content="line1\nline2\nline3\n",
        )
        p2 = FilePatch(
            file_path="b.py",
            original_content="x\ny\nz\n",
            patched_content="x\n",
        )
        proposal = self._make_proposal([p1, p2])
        added, removed = proposal.total_changes()
        assert added == p1.lines_added + p2.lines_added
        assert removed == p1.lines_removed + p2.lines_removed

    def test_as_diff_joins_patches(self):
        p1 = FilePatch(file_path="a.py", original_content="a\n", patched_content="b\n")
        p2 = FilePatch(file_path="c.py", original_content="c\n", patched_content="d\n")
        proposal = self._make_proposal([p1, p2])
        diff = proposal.as_diff()
        assert "a/a.py" in diff
        assert "a/c.py" in diff

    def test_apply_all_sets_status_applied(self, tmp_path: Path):
        p = FilePatch(file_path="f.py", original_content="old", patched_content="new")
        proposal = self._make_proposal([p])

        result = proposal.apply_all(tmp_path)

        assert result is True
        assert proposal.status == PatchStatus.APPLIED
        assert (tmp_path / "f.py").read_text() == "new"

    def test_apply_all_returns_false_if_patch_fails(self, tmp_path: Path):
        blocker = tmp_path / "blocker"
        blocker.write_text("file")

        good = FilePatch(file_path="ok.py", original_content="", patched_content="ok")
        bad = FilePatch(file_path="blocker/child/f.py", original_content="", patched_content="x")
        proposal = self._make_proposal([bad, good])

        result = proposal.apply_all(tmp_path)
        assert result is False
        # Status should NOT be APPLIED since it failed
        assert proposal.status != PatchStatus.APPLIED

    def test_revert_all(self, tmp_path: Path):
        target = tmp_path / "f.py"
        target.write_text("patched")

        p = FilePatch(file_path="f.py", original_content="original", patched_content="patched")
        proposal = self._make_proposal([p])

        result = proposal.revert_all(tmp_path)
        assert result is True
        assert target.read_text() == "original"

    def test_revert_all_returns_false_on_failure(self):
        p = FilePatch(file_path="f.py", original_content="x", patched_content="y")
        proposal = self._make_proposal([p])

        result = proposal.revert_all(Path("/nonexistent_path_for_testing_12345"))
        assert result is False


# ===========================================================================
# SimpleCodeGenerator tests
# ===========================================================================


class TestSimpleCodeGenerator:
    """Tests for SimpleCodeGenerator."""

    @pytest.fixture
    def gen(self) -> SimpleCodeGenerator:
        return SimpleCodeGenerator()

    # --- generate_fix ---

    @pytest.mark.asyncio
    async def test_generate_fix_test_async_missing_async_await(self, gen: SimpleCodeGenerator):
        """When category is TEST_ASYNC and content lacks async def / await."""
        analysis = _make_analysis(category=FailureCategory.TEST_ASYNC)
        content = "def test_foo():\n    result = some_coroutine()\n"

        fixed, rationale, confidence = await gen.generate_fix(analysis, content, "test.py")

        # The simple generator returns original content unchanged but sets rationale
        assert fixed == content
        assert "async" in rationale.lower() or "asyncio" in rationale.lower()
        assert confidence == 0.6

    @pytest.mark.asyncio
    async def test_generate_fix_test_async_already_has_async(self, gen: SimpleCodeGenerator):
        """When content already has async def, no special rationale is set."""
        analysis = _make_analysis(category=FailureCategory.TEST_ASYNC)
        content = "async def test_foo():\n    await something()\n"

        fixed, rationale, confidence = await gen.generate_fix(analysis, content, "test.py")

        assert fixed == content
        # Rationale should be empty since condition not met
        assert rationale == ""
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_generate_fix_test_mock_with_magicmock(self, gen: SimpleCodeGenerator):
        """When category is TEST_MOCK and MagicMock is present."""
        analysis = _make_analysis(category=FailureCategory.TEST_MOCK)
        content = "from unittest.mock import MagicMock\nmock = MagicMock()\n"

        fixed, rationale, confidence = await gen.generate_fix(analysis, content, "test.py")

        assert fixed == content
        assert "mock" in rationale.lower() or "Mock" in rationale
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_generate_fix_test_mock_without_magicmock(self, gen: SimpleCodeGenerator):
        """When category is TEST_MOCK but no MagicMock in content."""
        analysis = _make_analysis(category=FailureCategory.TEST_MOCK)
        content = "def test_foo():\n    pass\n"

        fixed, rationale, confidence = await gen.generate_fix(analysis, content, "test.py")

        assert fixed == content
        # No MagicMock found so the mock branch doesn't set rationale
        assert rationale == ""

    @pytest.mark.asyncio
    async def test_generate_fix_other_category(self, gen: SimpleCodeGenerator):
        """For a category with no special handling, returns content unchanged."""
        analysis = _make_analysis(category=FailureCategory.IMPL_BUG)
        content = "def foo():\n    return 1\n"

        fixed, rationale, confidence = await gen.generate_fix(analysis, content, "foo.py")

        assert fixed == content
        assert rationale == ""
        assert confidence == 0.5

    # --- critique_fix ---

    @pytest.mark.asyncio
    async def test_critique_fix_no_changes(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        original = "same content"

        critique, is_ok = await gen.critique_fix(analysis, original, original, "rationale")

        assert critique == "No changes were made to the file."
        assert is_ok is False

    @pytest.mark.asyncio
    async def test_critique_fix_bad_pattern_wildcard_import(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        original = "import os\n"
        proposed = "import os\nfrom foo import *\n"

        critique, is_ok = await gen.critique_fix(analysis, original, proposed, "rationale")

        assert "wildcard" in critique.lower() or "import" in critique.lower()
        assert is_ok is False

    @pytest.mark.asyncio
    async def test_critique_fix_bad_pattern_bare_except(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        original = "pass\n"
        proposed = "try:\n    pass\nexcept:\n"

        critique, is_ok = await gen.critique_fix(analysis, original, proposed, "rationale")

        assert "except" in critique.lower() or "bare" in critique.lower()
        assert is_ok is False

    @pytest.mark.asyncio
    async def test_critique_fix_bad_pattern_blanket_type_ignore(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        original = "x = 1\n"
        proposed = "x = 1  # type: ignore\n"

        critique, is_ok = await gen.critique_fix(analysis, original, proposed, "rationale")

        assert is_ok is False

    @pytest.mark.asyncio
    async def test_critique_fix_clean(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        original = "x = 1\n"
        proposed = "x = 2\n"

        critique, is_ok = await gen.critique_fix(analysis, original, proposed, "rationale")

        assert critique == "Fix appears reasonable."
        assert is_ok is True

    @pytest.mark.asyncio
    async def test_critique_fix_bad_pattern_already_in_original(self, gen: SimpleCodeGenerator):
        """If the bad pattern was already in the original, it should not be flagged."""
        analysis = _make_analysis()
        original = "from foo import *\nx = 1\n"
        proposed = "from foo import *\nx = 2\n"

        critique, is_ok = await gen.critique_fix(analysis, original, proposed, "rationale")

        assert is_ok is True

    # --- synthesize_fixes ---

    @pytest.mark.asyncio
    async def test_synthesize_fixes_empty(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()

        content, rationale, confidence = await gen.synthesize_fixes(analysis, [], [])

        assert content == ""
        assert rationale == "No proposals to synthesize"
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_fixes_single(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        proposals = [("fixed content", "reason", 0.8)]

        content, rationale, confidence = await gen.synthesize_fixes(analysis, proposals, [])

        assert content == "fixed content"
        assert confidence == 0.8

    @pytest.mark.asyncio
    async def test_synthesize_fixes_picks_highest_confidence(self, gen: SimpleCodeGenerator):
        analysis = _make_analysis()
        proposals = [
            ("low", "low reason", 0.3),
            ("high", "high reason", 0.9),
            ("mid", "mid reason", 0.6),
        ]

        content, rationale, confidence = await gen.synthesize_fixes(
            analysis, proposals, ["critique"]
        )

        assert content == "high"
        assert confidence == 0.9
        assert "high reason" in rationale


# ===========================================================================
# PatchProposer tests
# ===========================================================================


class TestPatchProposer:
    """Tests for PatchProposer."""

    @pytest.mark.asyncio
    async def test_propose_fix_file_not_found(self, tmp_path: Path):
        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="tests/nonexistent.py",
        )

        proposer = PatchProposer(repo_path=tmp_path)
        result = await proposer.propose_fix(analysis)

        assert result.status == PatchStatus.REJECTED
        assert "not found" in result.description.lower()

    @pytest.mark.asyncio
    async def test_propose_fix_reads_test_file_when_fix_target_test(self, tmp_path: Path):
        """When fix_target is TEST_FILE, propose_fix reads the test_file."""
        test_file = tmp_path / "tests" / "test_example.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test_foo():\n    assert True\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="tests/test_example.py",
        )

        proposer = PatchProposer(repo_path=tmp_path)
        result = await proposer.propose_fix(analysis)

        assert result.status == PatchStatus.SYNTHESIZED
        assert result.id == "fix_1"
        assert result.proposer == "hegelian_debate"

    @pytest.mark.asyncio
    async def test_propose_fix_reads_impl_file_when_fix_target_impl(self, tmp_path: Path):
        """When fix_target is IMPL_FILE, propose_fix reads root_cause_file."""
        impl_file = tmp_path / "src" / "example.py"
        impl_file.parent.mkdir(parents=True)
        impl_file.write_text("def foo():\n    return 1\n")

        analysis = _make_analysis(
            fix_target=FixTarget.IMPL_FILE,
            root_cause_file="src/example.py",
        )

        proposer = PatchProposer(repo_path=tmp_path)
        result = await proposer.propose_fix(analysis)

        assert result.status == PatchStatus.SYNTHESIZED

    @pytest.mark.asyncio
    async def test_propose_fix_single_generator(self, tmp_path: Path):
        """Single generator that actually changes the content produces a patch."""
        test_file = tmp_path / "test_f.py"
        test_file.write_text("original\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        mock_gen = AsyncMock()
        mock_gen.generate_fix = AsyncMock(return_value=("fixed\n", "changed it", 0.8))
        mock_gen.critique_fix = AsyncMock(return_value=("looks good", True))
        mock_gen.synthesize_fixes = AsyncMock(return_value=("fixed\n", "best fix", 0.8))

        proposer = PatchProposer(repo_path=tmp_path, generators=[mock_gen])
        result = await proposer.propose_fix(analysis)

        assert result.status == PatchStatus.SYNTHESIZED
        assert len(result.patches) == 1
        assert result.patches[0].patched_content == "fixed\n"
        assert result.post_debate_confidence == 0.8
        # Single generator means no cross-critique (skip self-critique)
        mock_gen.critique_fix.assert_not_called()

    @pytest.mark.asyncio
    async def test_propose_fix_multiple_generators_cross_critique(self, tmp_path: Path):
        """Multiple generators cross-critique each other."""
        test_file = tmp_path / "test_f.py"
        test_file.write_text("original\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        gen1 = AsyncMock()
        gen1.generate_fix = AsyncMock(return_value=("fix_a\n", "approach a", 0.7))
        gen1.critique_fix = AsyncMock(return_value=("critique from gen1", True))

        gen2 = AsyncMock()
        gen2.generate_fix = AsyncMock(return_value=("fix_b\n", "approach b", 0.9))
        gen2.critique_fix = AsyncMock(return_value=("critique from gen2", True))

        synthesizer = AsyncMock()
        synthesizer.synthesize_fixes = AsyncMock(return_value=("fix_b\n", "picked b", 0.9))

        proposer = PatchProposer(
            repo_path=tmp_path,
            generators=[gen1, gen2],
            synthesizer=synthesizer,
        )
        result = await proposer.propose_fix(analysis)

        assert result.status == PatchStatus.SYNTHESIZED
        # gen1 critiques gen2's proposal, gen2 critiques gen1's proposal
        assert gen1.critique_fix.call_count == 1
        assert gen2.critique_fix.call_count == 1
        assert len(result.critiques) == 2

    @pytest.mark.asyncio
    async def test_propose_fix_require_consensus_rejection(self, tmp_path: Path):
        """When require_consensus=True and critics reject, proposal is REJECTED."""
        test_file = tmp_path / "test_f.py"
        test_file.write_text("original\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        gen1 = AsyncMock()
        gen1.generate_fix = AsyncMock(return_value=("fix_a\n", "approach a", 0.7))
        gen1.critique_fix = AsyncMock(return_value=("bad fix", False))

        gen2 = AsyncMock()
        gen2.generate_fix = AsyncMock(return_value=("fix_b\n", "approach b", 0.5))
        gen2.critique_fix = AsyncMock(return_value=("also bad", False))

        synthesizer = AsyncMock()
        synthesizer.synthesize_fixes = AsyncMock(return_value=("fix_a\n", "synth", 0.6))

        proposer = PatchProposer(
            repo_path=tmp_path,
            generators=[gen1, gen2],
            synthesizer=synthesizer,
            require_consensus=True,
        )
        result = await proposer.propose_fix(analysis)

        assert result.status == PatchStatus.REJECTED
        assert "consensus" in result.description.lower()

    @pytest.mark.asyncio
    async def test_propose_fix_require_consensus_passes_with_majority(self, tmp_path: Path):
        """Consensus passes when majority of critiques approve."""
        test_file = tmp_path / "test_f.py"
        test_file.write_text("original\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        gen1 = AsyncMock()
        gen1.generate_fix = AsyncMock(return_value=("fix\n", "reason", 0.8))
        gen1.critique_fix = AsyncMock(return_value=("good", True))

        gen2 = AsyncMock()
        gen2.generate_fix = AsyncMock(return_value=("fix\n", "reason", 0.7))
        gen2.critique_fix = AsyncMock(return_value=("not great", False))

        synthesizer = AsyncMock()
        synthesizer.synthesize_fixes = AsyncMock(return_value=("fix\n", "synth", 0.8))

        proposer = PatchProposer(
            repo_path=tmp_path,
            generators=[gen1, gen2],
            synthesizer=synthesizer,
            require_consensus=True,
        )
        result = await proposer.propose_fix(analysis)

        # 1 approve, 1 reject => 1 >= 2//2 = 1, so consensus is True
        assert result.status == PatchStatus.SYNTHESIZED

    @pytest.mark.asyncio
    async def test_propose_fix_generator_exception_handled(self, tmp_path: Path):
        """If a generator raises an exception, the proposal still proceeds."""
        test_file = tmp_path / "test_f.py"
        test_file.write_text("original\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        failing_gen = AsyncMock()
        failing_gen.generate_fix = AsyncMock(side_effect=RuntimeError("boom"))
        failing_gen.critique_fix = AsyncMock(return_value=("ok", True))
        failing_gen.synthesize_fixes = AsyncMock(return_value=("original\n", "fallback", 0.0))

        proposer = PatchProposer(repo_path=tmp_path, generators=[failing_gen])
        result = await proposer.propose_fix(analysis)

        # Should still return a SYNTHESIZED result (no crash)
        assert result.status == PatchStatus.SYNTHESIZED

    @pytest.mark.asyncio
    async def test_propose_fix_no_change_produces_empty_patches(self, tmp_path: Path):
        """When the final content equals original, patches list is empty."""
        test_file = tmp_path / "test_f.py"
        original = "def test():\n    pass\n"
        test_file.write_text(original)

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        proposer = PatchProposer(repo_path=tmp_path)
        result = await proposer.propose_fix(analysis)

        # SimpleCodeGenerator returns content unchanged for TEST_ASSERTION
        assert result.patches == []

    @pytest.mark.asyncio
    async def test_proposal_counter_increments(self, tmp_path: Path):
        test_file = tmp_path / "test_f.py"
        test_file.write_text("content\n")

        analysis = _make_analysis(
            fix_target=FixTarget.TEST_FILE,
            test_file="test_f.py",
        )

        proposer = PatchProposer(repo_path=tmp_path)
        r1 = await proposer.propose_fix(analysis)
        r2 = await proposer.propose_fix(analysis)

        assert r1.id == "fix_1"
        assert r2.id == "fix_2"

    @pytest.mark.asyncio
    async def test_default_generators(self, tmp_path: Path):
        """With no generators specified, defaults to SimpleCodeGenerator."""
        proposer = PatchProposer(repo_path=tmp_path)
        assert len(proposer.generators) == 1
        assert isinstance(proposer.generators[0], SimpleCodeGenerator)
        assert proposer.synthesizer is proposer.generators[0]


# ===========================================================================
# ProposalDebate tests
# ===========================================================================


class TestProposalDebate:
    """Tests for the ProposalDebate dataclass."""

    def test_creation(self):
        proposal = PatchProposal(id="p1", analysis=_make_analysis())
        debate = ProposalDebate(proposal=proposal)

        assert debate.proposal is proposal
        assert debate.proposals == []
        assert debate.critiques == []
        assert debate.synthesis == ""
        assert debate.final_proposal is None
        assert debate.consensus_reached is False
        assert debate.dissenting_opinions == []
