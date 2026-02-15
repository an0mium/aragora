"""Tests for autonomous PR review in verify phase.

Verifies that:
- PRReviewRunner.review_diff() works without a GitHub URL
- VerifyPhase integrates PR review as a verification step
- Critical findings block the pipeline
- Errors in PR review don't block the pipeline
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.compat.openclaw.pr_review_runner import (
    PRReviewRunner,
    ReviewFinding,
    ReviewResult,
)


SAMPLE_DIFF = """\
diff --git a/aragora/server/handler.py b/aragora/server/handler.py
index abc1234..def5678 100644
--- a/aragora/server/handler.py
+++ b/aragora/server/handler.py
@@ -10,6 +10,8 @@
 def handle_request(req):
-    return process(req)
+    data = req.get_json()
+    result = process(data)
+    return result
"""


class TestReviewDiff:
    """Tests for PRReviewRunner.review_diff()."""

    @pytest.mark.asyncio
    async def test_returns_result_without_github_url(self):
        """review_diff should work without a GitHub PR URL."""
        runner = PRReviewRunner(dry_run=True, demo=True)

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = (
                {"agreement_score": 0.85, "agents_used": ["a", "b"]},
                None,
            )
            result = await runner.review_diff("some diff text")

        assert isinstance(result, ReviewResult)
        assert result.pr_url == "local"
        assert result.pr_number is None
        assert result.comment_posted is False
        assert result.receipt is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_empty_diff_returns_error(self):
        """Empty diff should return an error result."""
        runner = PRReviewRunner(dry_run=True)
        result = await runner.review_diff("")

        assert result.error == "Empty diff"
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_whitespace_only_diff_returns_error(self):
        """Whitespace-only diff should return an error result."""
        runner = PRReviewRunner(dry_run=True)
        result = await runner.review_diff("   \n  \t  ")

        assert result.error == "Empty diff"

    @pytest.mark.asyncio
    async def test_custom_label_in_receipt(self):
        """Label should appear in the receipt pr_url field."""
        runner = PRReviewRunner(dry_run=True, demo=True)

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = ({"agreement_score": 0.9}, None)
            result = await runner.review_diff(SAMPLE_DIFF, label="nomic-cycle-5")

        assert result.pr_url == "nomic-cycle-5"
        assert result.receipt is not None
        assert result.receipt.pr_url == "nomic-cycle-5"

    @pytest.mark.asyncio
    async def test_findings_parsed_from_review(self):
        """Findings from _run_review should be parsed into ReviewFinding objects."""
        runner = PRReviewRunner(dry_run=True, demo=True)

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = (
                {
                    "critical_issues": ["SQL injection in query builder"],
                    "high_issues": ["Missing auth check"],
                    "medium_issues": [],
                    "low_issues": ["Typo in variable name"],
                    "agreement_score": 0.75,
                },
                None,
            )
            result = await runner.review_diff(SAMPLE_DIFF)

        assert len(result.findings) == 3
        assert result.critical_count == 1
        assert result.high_count == 1
        assert result.has_critical is True

    @pytest.mark.asyncio
    async def test_no_comment_posted(self):
        """review_diff should never post GitHub comments."""
        runner = PRReviewRunner(dry_run=False, demo=True)  # dry_run=False

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = ({"agreement_score": 1.0}, None)
            result = await runner.review_diff(SAMPLE_DIFF)

        assert result.comment_posted is False
        assert result.comment_url is None

    @pytest.mark.asyncio
    async def test_review_failure_returns_error_result(self):
        """If _run_review fails, should return error result."""
        runner = PRReviewRunner(dry_run=True, demo=True)

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = (None, "aragora CLI not found")
            result = await runner.review_diff(SAMPLE_DIFF)

        assert result.error == "Review failed: aragora CLI not found"
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_diff_size_enforcement(self):
        """Oversized diffs should be truncated per policy."""
        runner = PRReviewRunner(dry_run=True, demo=True)
        runner.policy.max_diff_size_kb = 1  # 1KB limit

        big_diff = "+" * 2048  # 2KB

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = ({"agreement_score": 0.5}, None)
            await runner.review_diff(big_diff)

        # _run_review should have received truncated diff
        called_diff = mock_review.call_args[0][0]
        assert len(called_diff) <= 1024

    @pytest.mark.asyncio
    async def test_receipt_checksum_present(self):
        """Receipt should have a SHA-256 checksum."""
        runner = PRReviewRunner(dry_run=True, demo=True)

        with patch.object(runner, "_run_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = (
                {"low_issues": ["minor style issue"], "agreement_score": 0.9},
                None,
            )
            result = await runner.review_diff(SAMPLE_DIFF)

        assert result.receipt is not None
        assert len(result.receipt.checksum) == 64  # SHA-256 hex


class TestVerifyPhasePRReview:
    """Tests for VerifyPhase PR review integration."""

    def _make_verify_phase(self, **kwargs):
        """Create a VerifyPhase with PR review enabled."""
        from scripts.nomic.phases.verify import VerifyPhase

        return VerifyPhase(
            aragora_path=Path("/tmp/test-aragora"),
            enable_pr_review=kwargs.pop("enable_pr_review", True),
            pr_review_runner=kwargs.pop("pr_review_runner", None),
            log_fn=lambda msg: None,
            stream_emit_fn=lambda *args: None,
            record_replay_fn=lambda *args: None,
            save_state_fn=lambda state: None,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_pr_review_step_runs_when_enabled(self):
        """PR review should run as a verification step when enabled."""
        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(
            return_value=ReviewResult(
                pr_url="nomic-cycle-0",
                pr_number=None,
                repo=None,
                findings=[],
                agreement_score=0.95,
                agents_used=["a"],
                comment_posted=False,
                comment_url=None,
                receipt=None,
            )
        )

        phase = self._make_verify_phase(pr_review_runner=mock_runner)

        # Mock the prerequisite checks so they pass
        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_get_diff_text", new_callable=AsyncMock) as mock_diff,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}
            mock_diff.return_value = SAMPLE_DIFF

            result = await phase.execute()

        mock_runner.review_diff.assert_called_once()
        assert result["success"] is True
        # PR review check should be in the checks list
        check_names = [c["check"] for c in result["data"]["checks"]]
        assert "pr_review" in check_names

    @pytest.mark.asyncio
    async def test_critical_findings_block_pipeline(self):
        """Critical findings should cause the verify phase to fail."""
        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(
            return_value=ReviewResult(
                pr_url="nomic-cycle-0",
                pr_number=None,
                repo=None,
                findings=[
                    ReviewFinding(
                        severity="critical",
                        title="RCE vulnerability",
                        description="Remote code execution via eval()",
                    ),
                ],
                agreement_score=0.9,
                agents_used=["a"],
                comment_posted=False,
                comment_url=None,
                receipt=None,
            )
        )

        phase = self._make_verify_phase(pr_review_runner=mock_runner)

        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_get_diff_text", new_callable=AsyncMock) as mock_diff,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}
            mock_diff.return_value = SAMPLE_DIFF

            result = await phase.execute()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_non_critical_findings_dont_block(self):
        """High/medium/low findings should not block the pipeline."""
        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(
            return_value=ReviewResult(
                pr_url="nomic-cycle-0",
                pr_number=None,
                repo=None,
                findings=[
                    ReviewFinding(
                        severity="high",
                        title="Missing auth check",
                        description="Handler lacks authentication",
                    ),
                    ReviewFinding(
                        severity="medium",
                        title="Perf issue",
                        description="N+1 query",
                    ),
                ],
                agreement_score=0.8,
                agents_used=["a"],
                comment_posted=False,
                comment_url=None,
                receipt=None,
            )
        )

        phase = self._make_verify_phase(pr_review_runner=mock_runner)

        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_get_diff_text", new_callable=AsyncMock) as mock_diff,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}
            mock_diff.return_value = SAMPLE_DIFF

            result = await phase.execute()

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_pr_review_disabled_by_default(self):
        """PR review should not run when enable_pr_review=False."""
        phase = self._make_verify_phase(enable_pr_review=False)

        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_pr_review", new_callable=AsyncMock) as mock_pr,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}

            await phase.execute()

        mock_pr.assert_not_called()

    @pytest.mark.asyncio
    async def test_pr_review_error_doesnt_block(self):
        """PR review errors should not block the pipeline."""
        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock(side_effect=RuntimeError("CLI crashed"))

        phase = self._make_verify_phase(pr_review_runner=mock_runner)

        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_get_diff_text", new_callable=AsyncMock) as mock_diff,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}
            mock_diff.return_value = SAMPLE_DIFF

            result = await phase.execute()

        assert result["success"] is True
        pr_check = next(c for c in result["data"]["checks"] if c["check"] == "pr_review")
        assert pr_check["passed"] is True
        assert "error" in pr_check

    @pytest.mark.asyncio
    async def test_empty_diff_skips_review(self):
        """No diff should result in no PR review check."""
        mock_runner = MagicMock()
        mock_runner.review_diff = AsyncMock()

        phase = self._make_verify_phase(pr_review_runner=mock_runner)

        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_get_diff_text", new_callable=AsyncMock) as mock_diff,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}
            mock_diff.return_value = ""

            result = await phase.execute()

        mock_runner.review_diff.assert_not_called()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_creates_runner_lazily_when_not_injected(self):
        """When no runner is injected, one should be created on demand."""
        phase = self._make_verify_phase(pr_review_runner=None)

        mock_runner_instance = MagicMock()
        mock_runner_instance.review_diff = AsyncMock(
            return_value=ReviewResult(
                pr_url="nomic-cycle-0",
                pr_number=None,
                repo=None,
                findings=[],
                agreement_score=0.9,
                agents_used=["a"],
                comment_posted=False,
                comment_url=None,
                receipt=None,
            )
        )

        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
            patch.object(phase, "_get_diff_text", new_callable=AsyncMock) as mock_diff,
            patch(
                "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
                return_value=mock_runner_instance,
            ) as mock_cls,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {"check": "tests", "passed": True}
            mock_diff.return_value = SAMPLE_DIFF

            result = await phase.execute()

        mock_cls.assert_called_once_with(dry_run=True, demo=True)
        mock_runner_instance.review_diff.assert_called_once()
        assert result["success"] is True
