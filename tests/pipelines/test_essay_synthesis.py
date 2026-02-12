"""Tests for aragora.pipelines.essay_synthesis module."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

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
    SynthesisConfig,
    TopicCluster,
    AttributedClaim,
    EssaySection,
    EssayOutline,
    EssaySynthesisPipeline,
    create_essay_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(text="Test claim about systems thinking and complexity",
                confidence=0.8, claim_type="assertion", topics=None):
    return ClaimExtraction(
        claim=text,
        context="From a conversation",
        conversation_id="conv_001",
        confidence=confidence,
        claim_type=claim_type,
        topics=topics or [],
    )


def _make_evidence(eid="ev1", title="Paper Title", content="Evidence content about systems"):
    return Evidence(
        id=eid,
        source_type=SourceType.EXTERNAL_API,
        source_id="https://example.com",
        content=content,
        title=title,
    )


# ---------------------------------------------------------------------------
# SynthesisConfig
# ---------------------------------------------------------------------------

class TestSynthesisConfig:

    def test_defaults(self):
        cfg = SynthesisConfig()
        assert cfg.min_claim_length == 50
        assert cfg.max_claims_per_conversation == 20
        assert cfg.claim_confidence_threshold == 0.5
        assert cfg.min_cluster_size == 3
        assert cfg.max_clusters == 20
        assert cfg.similarity_threshold == 0.3
        assert cfg.target_word_count == 50000
        assert cfg.section_word_target == 5000
        assert cfg.max_sections == 15
        assert cfg.include_counterarguments is True
        assert cfg.include_methodology is True

    def test_custom_values(self):
        cfg = SynthesisConfig(min_claim_length=100, max_clusters=5)
        assert cfg.min_claim_length == 100
        assert cfg.max_clusters == 5


# ---------------------------------------------------------------------------
# TopicCluster
# ---------------------------------------------------------------------------

class TestTopicCluster:

    def test_claim_count(self):
        claims = [_make_claim(f"Claim {i}") for i in range(5)]
        cluster = TopicCluster(
            id="c1", name="Test", description="Desc",
            keywords=["test"], claims=claims,
        )
        assert cluster.claim_count == 5

    def test_average_confidence(self):
        claims = [_make_claim(confidence=0.6), _make_claim(confidence=0.8)]
        cluster = TopicCluster(
            id="c1", name="Test", description="Desc",
            keywords=["test"], claims=claims,
        )
        assert abs(cluster.average_confidence - 0.7) < 0.01

    def test_average_confidence_empty(self):
        cluster = TopicCluster(
            id="c1", name="Test", description="Desc",
            keywords=["test"], claims=[],
        )
        assert cluster.average_confidence == 0.0

    def test_to_dict(self):
        claim = _make_claim()
        cluster = TopicCluster(
            id="c1", name="Test", description="Desc",
            keywords=["test"], claims=[claim],
            coherence_score=0.5, representative_claim=claim,
        )
        d = cluster.to_dict()
        assert d["id"] == "c1"
        assert d["claim_count"] == 1
        assert d["coherence_score"] == 0.5
        assert d["representative_claim"] is not None

    def test_to_dict_no_representative(self):
        cluster = TopicCluster(
            id="c1", name="Test", description="Desc",
            keywords=["test"], claims=[],
        )
        d = cluster.to_dict()
        assert d["representative_claim"] is None


# ---------------------------------------------------------------------------
# AttributedClaim
# ---------------------------------------------------------------------------

class TestAttributedClaim:

    def test_to_dict(self):
        claim = _make_claim()
        ev = _make_evidence()
        ac = AttributedClaim(
            claim=claim,
            sources=[ev],
            attribution_confidence=0.8,
            supporting_quotes=["A good quote"],
            synthesis_notes="Note",
            scholarly_context="Context",
        )
        d = ac.to_dict()
        assert d["attribution_confidence"] == 0.8
        assert len(d["sources"]) == 1
        assert d["supporting_quotes"] == ["A good quote"]
        assert d["contradicting_source_count"] == 0

    def test_defaults(self):
        claim = _make_claim()
        ac = AttributedClaim(claim=claim, sources=[], attribution_confidence=0.0)
        assert ac.supporting_quotes == []
        assert ac.contradicting_sources == []
        assert ac.synthesis_notes == ""


# ---------------------------------------------------------------------------
# EssaySection
# ---------------------------------------------------------------------------

class TestEssaySection:

    def test_to_dict(self):
        section = EssaySection(
            id="s1", title="Intro", level=1, content="Hello",
            word_count=100, claims_referenced=["c1"], sources_cited=["s1"],
        )
        d = section.to_dict()
        assert d["id"] == "s1"
        assert d["level"] == 1
        assert d["word_count"] == 100
        assert d["subsections"] == []

    def test_subsections(self):
        sub = EssaySection(
            id="s1_a", title="Sub", level=2, content="Sub content",
            word_count=50, claims_referenced=[], sources_cited=[],
        )
        parent = EssaySection(
            id="s1", title="Parent", level=1, content="Content",
            word_count=100, claims_referenced=[], sources_cited=[],
            subsections=[sub],
        )
        d = parent.to_dict()
        assert len(d["subsections"]) == 1
        assert d["subsections"][0]["id"] == "s1_a"


# ---------------------------------------------------------------------------
# EssayOutline
# ---------------------------------------------------------------------------

class TestEssayOutline:

    def test_total_words(self):
        section = EssaySection(
            id="s1", title="A", level=1, content="",
            word_count=100, claims_referenced=[], sources_cited=[],
        )
        outline = EssayOutline(
            title="Test", thesis="Thesis", sections=[section],
            target_word_count=50000,
        )
        assert outline.total_words == 100

    def test_total_words_with_subsections(self):
        sub = EssaySection(
            id="s1_a", title="Sub", level=2, content="",
            word_count=50, claims_referenced=[], sources_cited=[],
        )
        section = EssaySection(
            id="s1", title="A", level=1, content="",
            word_count=100, claims_referenced=[], sources_cited=[],
            subsections=[sub],
        )
        outline = EssayOutline(
            title="Test", thesis="Thesis", sections=[section],
            target_word_count=50000,
        )
        assert outline.total_words == 150

    def test_section_count(self):
        sub = EssaySection(
            id="s1_a", title="Sub", level=2, content="",
            word_count=50, claims_referenced=[], sources_cited=[],
        )
        section = EssaySection(
            id="s1", title="A", level=1, content="",
            word_count=100, claims_referenced=[], sources_cited=[],
            subsections=[sub],
        )
        outline = EssayOutline(
            title="Test", thesis="Thesis", sections=[section],
            target_word_count=50000,
        )
        assert outline.section_count == 2

    def test_to_dict(self):
        outline = EssayOutline(
            title="Test Essay", thesis="Main thesis",
            sections=[], target_word_count=50000,
        )
        d = outline.to_dict()
        assert d["title"] == "Test Essay"
        assert d["thesis"] == "Main thesis"
        assert d["total_words"] == 0
        assert d["section_count"] == 0
        assert "generated_at" in d


# ---------------------------------------------------------------------------
# EssaySynthesisPipeline
# ---------------------------------------------------------------------------

class TestEssaySynthesisPipeline:

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_default_config(self, mock_ingestor):
        pipeline = EssaySynthesisPipeline()
        assert isinstance(pipeline.config, SynthesisConfig)
        assert pipeline.connectors == {}

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_custom_config(self, mock_ingestor):
        cfg = SynthesisConfig(min_claim_length=100)
        pipeline = EssaySynthesisPipeline(config=cfg)
        assert pipeline.config.min_claim_length == 100

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_load_conversations_file(self, mock_ingestor_cls, tmp_path):
        mock_ingestor = MagicMock()
        mock_ingestor_cls.return_value = mock_ingestor
        export = MagicMock(spec=ConversationExport)
        mock_ingestor.load_export.return_value = export

        pipeline = EssaySynthesisPipeline()
        f = tmp_path / "export.json"
        f.write_text("{}")

        result = pipeline.load_conversations(f)
        assert result == [export]
        mock_ingestor.load_export.assert_called_once()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_load_conversations_directory(self, mock_ingestor_cls, tmp_path):
        mock_ingestor = MagicMock()
        mock_ingestor_cls.return_value = mock_ingestor
        mock_ingestor.load_directory.return_value = [MagicMock(), MagicMock()]

        pipeline = EssaySynthesisPipeline()
        result = pipeline.load_conversations(tmp_path)
        assert len(result) == 2

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_load_conversations_missing_path(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.load_conversations("/nonexistent/path")

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_deduplicate_claims(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        claims = [
            _make_claim("Systems thinking is fundamental to understanding complexity"),
            _make_claim("Systems thinking is fundamental to understanding complexity"),
            _make_claim("A different claim about evolution and adaptation patterns"),
        ]
        unique = pipeline._deduplicate_claims(claims)
        assert len(unique) == 2

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_extract_keywords(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        keywords = pipeline._extract_keywords(
            "The systems thinking approach reveals important complexity patterns"
        )
        assert "systems" in keywords
        assert "complexity" in keywords
        assert "patterns" in keywords
        # Stopwords excluded
        assert "the" not in keywords

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_cluster_claims_empty_raises(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        with pytest.raises(ValueError, match="No claims to cluster"):
            pipeline.cluster_claims()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_calculate_coherence_single_claim(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        claims = [_make_claim("single claim")]
        keywords = {claims[0].claim: {"single", "claim"}}
        score = pipeline._calculate_coherence(claims, keywords)
        assert score == 0.0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_calculate_attribution_confidence_no_supporting(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        confidence = pipeline._calculate_attribution_confidence([], [])
        assert confidence == 0.0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_calculate_attribution_confidence_with_sources(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        sources = [_make_evidence(f"ev{i}") for i in range(3)]
        confidence = pipeline._calculate_attribution_confidence(sources, [])
        assert 0.0 < confidence <= 1.0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_calculate_attribution_confidence_with_contradictions(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        supporting = [_make_evidence("ev1")]
        contradicting = [_make_evidence(f"ev{i}") for i in range(5)]
        conf = pipeline._calculate_attribution_confidence(supporting, contradicting)
        # Should be lower due to contradiction penalty
        no_contra_conf = pipeline._calculate_attribution_confidence(supporting, [])
        assert conf < no_contra_conf

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_thesis_empty_clusters(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        thesis = pipeline._generate_thesis([])
        assert "intellectual discourse" in thesis

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_thesis_with_clusters(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        clusters = [
            TopicCluster(
                id="c1", name="AI", description="AI cluster",
                keywords=["intelligence", "artificial", "alignment"],
                claims=[],
            ),
            TopicCluster(
                id="c2", name="Systems", description="Systems cluster",
                keywords=["systems", "complexity"],
                claims=[],
            ),
        ]
        thesis = pipeline._generate_thesis(clusters)
        assert "intelligence" in thesis
        assert "synthesis" in thesis.lower()

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_get_statistics(self, mock_ingestor_cls):
        mock_ingestor = MagicMock()
        mock_ingestor_cls.return_value = mock_ingestor
        mock_ingestor.get_statistics.return_value = {"conversations": 5}

        pipeline = EssaySynthesisPipeline()
        stats = pipeline.get_statistics()
        assert stats["conversations"] == 5
        assert stats["claims_extracted"] == 0
        assert stats["clusters_created"] == 0

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_generate_search_queries(self, mock_ingestor_cls):
        pipeline = EssaySynthesisPipeline()
        claim = _make_claim(
            "Artificial intelligence systems exhibit instrumental convergence patterns",
            topics=["AI safety"],
        )
        queries = pipeline._generate_search_queries(claim)
        assert len(queries) >= 1
        assert queries[0] == claim.claim[:200]


class TestCreateEssayPipeline:

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_creates_pipeline(self, mock_ingestor):
        pipeline = create_essay_pipeline()
        assert isinstance(pipeline, EssaySynthesisPipeline)

    @patch("aragora.pipelines.essay_synthesis.ConversationIngestorConnector")
    def test_with_config(self, mock_ingestor):
        cfg = SynthesisConfig(target_word_count=10000)
        pipeline = create_essay_pipeline(config=cfg)
        assert pipeline.config.target_word_count == 10000
