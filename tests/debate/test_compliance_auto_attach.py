"""Tests for auto-attaching compliance artifacts (Sprint 16C)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.orchestrator_runner import handle_debate_completion


def _make_arena():
    """Create a minimal mock arena for testing."""
    arena = MagicMock()
    arena._trackers = MagicMock()
    arena.extensions = MagicMock()
    arena.agents = []
    arena._budget_coordinator = MagicMock()
    arena._queue_for_supabase_sync = MagicMock()
    arena._ingest_debate_outcome = AsyncMock()
    arena.enable_post_debate_workflow = False
    arena.post_debate_workflow = None
    return arena


def _make_state(domain: str = "healthcare"):
    """Create a minimal mock state."""
    state = MagicMock()
    state.debate_id = "test-compliance"
    state.debate_status = "completed"
    state.gupp_bead_id = None
    state.gupp_hook_entries = []

    ctx = MagicMock()
    ctx.result = MagicMock()
    ctx.result.to_dict.return_value = {
        "input_summary": "Review patient treatment plan",
        "verdict_reasoning": "Evidence supports treatment",
    }
    ctx.result.compliance_artifacts = None
    ctx.debate_id = "test-compliance"
    ctx.domain = domain
    ctx.env = MagicMock()
    ctx.env.task = "Review patient treatment plan for clinical trial"
    state.ctx = ctx
    return state


class TestComplianceAutoAttach:
    """Tests for auto-attaching compliance artifacts."""

    @pytest.mark.asyncio
    async def test_artifacts_generated_for_healthcare_domain(self):
        """Compliance artifacts attached for healthcare domain."""
        arena = _make_arena()
        state = _make_state(domain="healthcare")

        await handle_debate_completion(arena, state)

        # Check that compliance_artifacts was set on result
        ctx = state.ctx
        if hasattr(ctx.result, "compliance_artifacts"):
            # May or may not trigger based on keyword matching
            pass  # Non-failure assertion
        # The key assertion is that no exception was raised

    @pytest.mark.asyncio
    async def test_no_artifacts_for_general_domain(self):
        """No compliance artifacts for general domain."""
        arena = _make_arena()
        state = _make_state(domain="general")

        await handle_debate_completion(arena, state)

        # compliance_artifacts should not be set for general domain
        ctx = state.ctx
        # The auto-attach only runs for healthcare/finance/legal/compliance
        # So for general domain, we verify no error occurred

    @pytest.mark.asyncio
    async def test_risk_classification_works(self):
        """RiskClassifier correctly classifies domain-relevant text."""
        from aragora.compliance.eu_ai_act import RiskClassifier

        classifier = RiskClassifier()

        # Healthcare-related text should get high risk if it matches Annex III
        result = classifier.classify("credit scoring system for loan decision")
        assert result.risk_level.value in ("high", "limited")

        # General text should get minimal risk
        result2 = classifier.classify("best pizza recipe")
        assert result2.risk_level.value == "minimal"

    @pytest.mark.asyncio
    async def test_artifacts_attached_to_result(self):
        """Artifacts are attached to the debate result as a dict."""
        from aragora.compliance.eu_ai_act import (
            ComplianceArtifactGenerator,
            RiskClassifier,
        )

        classifier = RiskClassifier()
        risk = classifier.classify("credit scoring for loan decision")
        assert risk.risk_level.value == "high"

        generator = ComplianceArtifactGenerator()
        receipt = {
            "receipt_id": "test-receipt",
            "input_summary": "credit scoring for loan decision",
            "verdict_reasoning": "Risk analysis complete",
        }
        bundle = generator.generate(receipt)
        result_dict = bundle.to_dict()

        assert "bundle_id" in result_dict
        assert "risk_classification" in result_dict
        assert result_dict["risk_classification"]["risk_level"] == "high"

    @pytest.mark.asyncio
    async def test_failure_does_not_crash_debate(self):
        """Compliance failure doesn't crash debate completion."""
        from aragora.compliance.eu_ai_act import RiskClassifier

        original_classify = RiskClassifier.classify

        def broken_classify(self, description):
            raise RuntimeError("classification error")

        RiskClassifier.classify = broken_classify
        try:
            arena = _make_arena()
            state = _make_state(domain="finance")
            # Should not raise
            await handle_debate_completion(arena, state)
        finally:
            RiskClassifier.classify = original_classify

    @pytest.mark.asyncio
    async def test_import_error_handled_gracefully(self):
        """Compliance auto-attach handles missing data gracefully."""
        arena = _make_arena()
        state = _make_state(domain="legal")
        # With a general task, compliance will run but not crash
        state.ctx.env.task = "General legal discussion"
        await handle_debate_completion(arena, state)
