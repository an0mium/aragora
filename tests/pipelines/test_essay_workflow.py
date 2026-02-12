"""Tests for aragora.pipelines.essay_workflow module."""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.conversation_ingestor import (
    ClaimExtraction,
    ConversationExport,
    Conversation,
    ConversationMessage,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType
from aragora.pipelines.essay_synthesis import (
    EssayOutline,
    EssaySection,
    SynthesisConfig,
    TopicCluster,
    AttributedClaim,
)
from aragora.pipelines.essay_workflow import (
    WorkflowConfig,
    SeedEssay,
    DebateResult,
    WorkflowResult,
    EssayWorkflow,
    create_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(text="Test claim about complex systems", confidence=0.8,
                claim_type="assertion", topics=None):
    return ClaimExtraction(
        claim=text, context="context", conversation_id="conv_1",
        confidence=confidence, claim_type=claim_type,
        topics=topics or [],
    )


def _make_evidence(eid="ev1"):
    return Evidence(
        id=eid, source_type=SourceType.EXTERNAL_API,
        source_id="https://example.com", content="Evidence about systems",
        title="Paper Title",
    )


# ---------------------------------------------------------------------------
# WorkflowConfig
# ---------------------------------------------------------------------------

class TestWorkflowConfig:

    def test_defaults(self):
        cfg = WorkflowConfig()
        assert cfg.min_claim_length == 50
        assert cfg.enable_debate is True
        assert cfg.debate_rounds == 9
        assert len(cfg.debate_agents) == 8
        assert cfg.enable_attribution is True
        assert cfg.target_word_count == 50000
        assert cfg.voice_style == "analytical"
        assert cfg.output_format == "markdown"
        assert cfg.enable_seed_filtering is True
        assert cfg.seed_min_score == 0.05

    def test_custom_values(self):
        cfg = WorkflowConfig(debate_rounds=3, voice_style="academic")
        assert cfg.debate_rounds == 3
        assert cfg.voice_style == "academic"


# ---------------------------------------------------------------------------
# SeedEssay
# ---------------------------------------------------------------------------

class TestSeedEssay:

    def test_from_markdown(self, tmp_path):
        md = tmp_path / "seed.md"
        md.write_text(
            "# My Essay Title\n\n"
            "## Introduction\n\nSome intro text.\n\n"
            "## Analysis\n\nI argue that complex systems exhibit emergent behavior.\n"
        )
        seed = SeedEssay.from_markdown(md)
        assert seed.title == "My Essay Title"
        assert len(seed.sections) >= 2
        assert "Introduction" in seed.themes or "My Essay Title" in seed.themes
        assert len(seed.keywords) > 0

    def test_from_markdown_no_h1(self, tmp_path):
        md = tmp_path / "untitled.md"
        md.write_text("Just some text without headings.")
        seed = SeedEssay.from_markdown(md)
        assert seed.title == "untitled"  # Falls back to filename stem

    def test_from_markdown_extracts_key_claims(self, tmp_path):
        md = tmp_path / "claims.md"
        md.write_text(
            "# Thesis Paper\n\n"
            "I argue that artificial intelligence will transform every industry "
            "through incremental automation rather than sudden revolution.\n"
        )
        seed = SeedEssay.from_markdown(md)
        assert len(seed.key_claims) >= 1

    def test_extract_keywords_static(self):
        keywords = SeedEssay._extract_keywords(
            "Systems thinking reveals complexity patterns in organizations"
        )
        assert "systems" in keywords
        assert "complexity" in keywords
        assert "the" not in keywords

    def test_defaults(self):
        seed = SeedEssay(title="Test", content="Content")
        assert seed.sections == []
        assert seed.key_claims == []
        assert seed.themes == []
        assert seed.keywords == set()
        assert seed.metadata == {}


# ---------------------------------------------------------------------------
# DebateResult
# ---------------------------------------------------------------------------

class TestDebateResult:

    def test_defaults(self):
        dr = DebateResult(original_claim="Claim")
        assert dr.strengthened_claim is None
        assert dr.counterarguments == []
        assert dr.rebuttals == []
        assert dr.consensus_reached is False
        assert dr.confidence_change == 0.0

    def test_to_dict(self):
        dr = DebateResult(
            original_claim="Claim",
            strengthened_claim="Better claim",
            counterarguments=["counter1"],
            consensus_reached=True,
        )
        d = dr.to_dict()
        assert d["original_claim"] == "Claim"
        assert d["strengthened_claim"] == "Better claim"
        assert d["consensus_reached"] is True


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

class TestWorkflowResult:

    def test_to_dict(self):
        outline = EssayOutline(
            title="Test", thesis="Thesis", sections=[],
            target_word_count=50000,
        )
        result = WorkflowResult(
            title="Test Essay",
            thesis="Main thesis",
            outline=outline,
            claims=[_make_claim()],
            clusters=[],
            attributed_claims=[],
            debate_results=[],
            steelman_claims={},
            seed_essay=None,
            final_essay="Final text",
            statistics={"total": 1},
        )
        d = result.to_dict()
        assert d["title"] == "Test Essay"
        assert d["claim_count"] == 1
        assert d["has_seed_essay"] is False
        assert "generated_at" in d


# ---------------------------------------------------------------------------
# EssayWorkflow
# ---------------------------------------------------------------------------

class TestEssayWorkflow:

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_default_config(self, mock_ingestor):
        wf = EssayWorkflow()
        assert isinstance(wf.config, WorkflowConfig)
        assert wf._seed_essay is None
        assert wf._exports == []

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_custom_config(self, mock_ingestor):
        cfg = WorkflowConfig(debate_rounds=3)
        wf = EssayWorkflow(config=cfg)
        assert wf.config.debate_rounds == 3

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_set_seed_essay_from_content(self, mock_ingestor):
        wf = EssayWorkflow()
        result = asyncio.get_event_loop().run_until_complete(
            wf.set_seed_essay(content="Some essay content about evolution", title="My Essay")
        )
        assert result is not None
        assert result.title == "My Essay"
        assert len(result.keywords) > 0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_set_seed_essay_from_file(self, mock_ingestor, tmp_path):
        md = tmp_path / "seed.md"
        md.write_text("# Test\n\nContent about systems thinking.")
        wf = EssayWorkflow()
        result = asyncio.get_event_loop().run_until_complete(
            wf.set_seed_essay(path=md)
        )
        assert result.title == "Test"

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_set_seed_essay_none(self, mock_ingestor):
        wf = EssayWorkflow()
        result = asyncio.get_event_loop().run_until_complete(
            wf.set_seed_essay()
        )
        assert result is None
        assert wf._seed_essay is None

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_counterarguments_absolute(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim("AI will always surpass human intelligence")
        counters = wf._generate_counterarguments(claim)
        assert any("absolute" in c.lower() for c in counters)

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_counterarguments_causal(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim("Social media causes depression in all teenagers")
        counters = wf._generate_counterarguments(claim)
        assert any("correlation" in c.lower() for c in counters)

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_counterarguments_predictive(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim("Artificial general intelligence will inevitably emerge")
        counters = wf._generate_counterarguments(claim)
        assert any("prediction" in c.lower() for c in counters)

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_counterarguments_generic(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim("Functional programming has unique properties for distributed computation")
        counters = wf._generate_counterarguments(claim)
        assert any("alternative" in c.lower() for c in counters)

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_rebuttals(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim()
        counters = ["Uses absolute language", "Correlation issue", "Prediction problem", "Other"]
        rebuttals = wf._generate_rebuttals(claim, counters)
        assert len(rebuttals) == 4

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_strengthen_claim_replaces_absolutes(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim("All systems always converge to equilibrium")
        counters = []
        rebuttals = []
        result = wf._strengthen_claim(claim, counters, rebuttals)
        assert "typically" in result
        assert "most" in result.lower()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_steelman_claim_analytical(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim("All things converge")
        result = wf._steelman_claim(claim)
        assert "stronger version" in result.lower()
        assert "boundary conditions" in result.lower()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_steelman_claim_academic(self, mock_ingestor):
        cfg = WorkflowConfig(voice_style="academic")
        wf = EssayWorkflow(config=cfg)
        claim = _make_claim("All things converge")
        result = wf._steelman_claim(claim)
        assert "defensible" in result.lower()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_steelman_claim_conversational(self, mock_ingestor):
        cfg = WorkflowConfig(voice_style="conversational")
        wf = EssayWorkflow(config=cfg)
        claim = _make_claim("Never underestimate complexity")
        result = wf._steelman_claim(claim)
        assert "stronger version" in result.lower()
        assert "practice" in result.lower()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_score_claim_by_seed_no_seed(self, mock_ingestor):
        wf = EssayWorkflow()
        claim = _make_claim()
        assert wf.score_claim_by_seed(claim) == 0.0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_score_claim_by_seed_with_seed(self, mock_ingestor):
        wf = EssayWorkflow()
        wf._seed_essay = SeedEssay(
            title="Test", content="systems complexity evolution",
            keywords={"systems", "complexity", "evolution"},
        )
        claim = _make_claim("Systems thinking reveals complexity patterns")
        score = wf.score_claim_by_seed(claim)
        assert score > 0.0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_filter_claims_by_seed_no_seed(self, mock_ingestor):
        wf = EssayWorkflow()
        claims = [_make_claim()]
        filtered, preview = wf.filter_claims_by_seed(claims)
        assert filtered == claims
        assert preview == []

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_export_essay_markdown(self, mock_ingestor, tmp_path):
        wf = EssayWorkflow()
        outline = EssayOutline(
            title="Test", thesis="Thesis", sections=[],
            target_word_count=50000,
        )
        result = WorkflowResult(
            title="Test", thesis="Thesis", outline=outline,
            claims=[], clusters=[], attributed_claims=[],
            debate_results=[], steelman_claims={}, seed_essay=None,
            final_essay="# Final Essay\n\nContent here.",
            statistics={},
        )
        out = tmp_path / "essay.md"
        wf.export_essay(out, result=result)
        assert out.read_text() == "# Final Essay\n\nContent here."

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_export_essay_json(self, mock_ingestor, tmp_path):
        cfg = WorkflowConfig(output_format="json")
        wf = EssayWorkflow(config=cfg)
        outline = EssayOutline(
            title="Test", thesis="Thesis", sections=[],
            target_word_count=50000,
        )
        result = WorkflowResult(
            title="Test", thesis="Thesis", outline=outline,
            claims=[], clusters=[], attributed_claims=[],
            debate_results=[], steelman_claims={}, seed_essay=None,
            final_essay="Content", statistics={},
        )
        out = tmp_path / "essay.json"
        wf.export_essay(out, result=result)
        data = json.loads(out.read_text())
        assert data["title"] == "Test"

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_export_essay_no_result_raises(self, mock_ingestor, tmp_path):
        wf = EssayWorkflow()
        with pytest.raises(ValueError, match="No result"):
            wf.export_essay(tmp_path / "out.md")

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_thread_skeleton(self, mock_ingestor):
        wf = EssayWorkflow()
        section = EssaySection(
            id="s1", title="AI and Evolution", level=1, content="",
            word_count=100, claims_referenced=[], sources_cited=[],
        )
        outline = EssayOutline(
            title="My Essay", thesis="Main argument about systems.",
            sections=[section], target_word_count=50000,
        )
        thread = wf.generate_thread_skeleton(outline, max_posts=5)
        assert "1/" in thread
        assert "My Essay" in thread


class TestCreateWorkflow:

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_creates_workflow(self, mock_ingestor):
        wf = create_workflow()
        assert isinstance(wf, EssayWorkflow)

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_with_config(self, mock_ingestor):
        cfg = WorkflowConfig(target_word_count=10000)
        wf = create_workflow(config=cfg)
        assert wf.config.target_word_count == 10000
