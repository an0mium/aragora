"""Tests for aragora.pipelines.prose_synthesis module."""

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipelines.prose_synthesis import (
    META_PATTERNS,
    clean_prose,
    ThemeConfig,
    DEFAULT_THEMES,
    ProsePassage,
    SynthesisResult,
    ProseSynthesisPipeline,
    create_prose_pipeline,
)


# ---------------------------------------------------------------------------
# META_PATTERNS regex tests
# ---------------------------------------------------------------------------

class TestMetaPatterns:

    def test_opening_acknowledgment_absolutely(self):
        assert any(
            re.match(p, "Absolutely! Here is the essay", re.IGNORECASE)
            for p in META_PATTERNS
        )

    def test_opening_acknowledgment_certainly(self):
        assert any(
            re.match(p, "Certainly, I'll help with that", re.IGNORECASE)
            for p in META_PATTERNS
        )

    def test_here_is_pattern(self):
        assert any(
            re.match(p, "Here's a detailed analysis", re.IGNORECASE)
            for p in META_PATTERNS
        )

    def test_let_me_pattern(self):
        assert any(
            re.match(p, "Let me help you with this", re.IGNORECASE)
            for p in META_PATTERNS
        )

    def test_horizontal_rule(self):
        assert any(
            re.match(p, "---", re.IGNORECASE)
            for p in META_PATTERNS
        )

    def test_note_prefix(self):
        assert any(
            re.match(p, "Note: this is important", re.IGNORECASE)
            for p in META_PATTERNS
        )

    def test_normal_text_not_matched(self):
        text = "The evolution of artificial intelligence poses unique challenges."
        matched = any(
            re.match(p, text.strip(), re.IGNORECASE)
            for p in META_PATTERNS
        )
        assert not matched


# ---------------------------------------------------------------------------
# clean_prose
# ---------------------------------------------------------------------------

class TestCleanProse:

    def test_removes_meta_commentary(self):
        text = "Absolutely! Here is the essay\n\nActual content here."
        cleaned = clean_prose(text)
        assert "Absolutely" not in cleaned
        assert "Actual content here." in cleaned

    def test_removes_inline_references(self):
        text = "This is important (as mentioned above) for understanding."
        cleaned = clean_prose(text)
        assert "(as mentioned above)" not in cleaned
        assert "This is important" in cleaned

    def test_removes_excessive_whitespace(self):
        text = "First paragraph.\n\n\n\n\n\nSecond paragraph."
        cleaned = clean_prose(text)
        assert "\n\n\n\n" not in cleaned

    def test_preserves_normal_content(self):
        text = "Intelligence is pattern rather than substance."
        cleaned = clean_prose(text)
        assert cleaned == text

    def test_removes_empty_headers(self):
        text = "# \nSome content after."
        cleaned = clean_prose(text)
        assert cleaned.startswith("Some content") or "# " not in cleaned


# ---------------------------------------------------------------------------
# ThemeConfig
# ---------------------------------------------------------------------------

class TestThemeConfig:

    def test_creation(self):
        tc = ThemeConfig(
            id="test", name="Test Theme", short_name="Test",
            narrative_position=1, keywords=["keyword1"],
        )
        assert tc.id == "test"
        assert tc.name == "Test Theme"
        assert tc.narrative_position == 1
        assert tc.keywords == ["keyword1"]
        assert tc.introduction == ""
        assert tc.transitions_to == {}

    def test_default_themes_exist(self):
        assert len(DEFAULT_THEMES) == 7
        assert "ai_risk_evolution" in DEFAULT_THEMES
        assert "systems_complexity" in DEFAULT_THEMES
        assert "religion_morality" in DEFAULT_THEMES
        assert "intelligence_consciousness" in DEFAULT_THEMES
        assert "rate_durability" in DEFAULT_THEMES
        assert "art_aesthetics" in DEFAULT_THEMES
        assert "politics_legitimacy" in DEFAULT_THEMES

    def test_default_themes_have_keywords(self):
        for theme_id, config in DEFAULT_THEMES.items():
            assert len(config.keywords) > 0, f"Theme {theme_id} has no keywords"

    def test_default_themes_narrative_positions(self):
        positions = [c.narrative_position for c in DEFAULT_THEMES.values()]
        assert sorted(positions) == list(range(1, 8))


# ---------------------------------------------------------------------------
# ProsePassage
# ---------------------------------------------------------------------------

class TestProsePassage:

    def test_creation(self):
        p = ProsePassage(
            text="Original text here",
            role="user",
            conversation_title="Chat about AI",
            quality_score=0.9,
            primary_theme="ai_risk_evolution",
            word_count=3,
        )
        assert p.text == "Original text here"
        assert p.role == "user"
        assert p.quality_score == 0.9

    def test_cleaned_text(self):
        p = ProsePassage(
            text="Absolutely! Here is content.\n\nActual prose here.",
            role="assistant",
            conversation_title="Chat",
            quality_score=0.8,
            primary_theme="ai_risk_evolution",
            word_count=10,
        )
        cleaned = p.cleaned_text
        assert "Absolutely" not in cleaned
        assert "Actual prose here." in cleaned

    def test_defaults(self):
        p = ProsePassage(
            text="Text", role="user", conversation_title="Conv",
            quality_score=0.5, primary_theme="test", word_count=1,
        )
        assert p.depth_score == 0.0
        assert p.novelty_score == 0.0
        assert p.beauty_score == 0.0


# ---------------------------------------------------------------------------
# SynthesisResult
# ---------------------------------------------------------------------------

class TestSynthesisResult:

    def test_to_dict(self):
        sr = SynthesisResult(
            title="Essay", content="Content here", word_count=2,
            themes_included=["ai_risk_evolution"], passages_used=5,
        )
        d = sr.to_dict()
        assert d["title"] == "Essay"
        assert d["word_count"] == 2
        assert d["themes_included"] == ["ai_risk_evolution"]
        assert d["passages_used"] == 5
        assert "generated_at" in d

    def test_defaults(self):
        sr = SynthesisResult(
            title="T", content="C", word_count=1,
            themes_included=[], passages_used=0,
        )
        assert sr.metadata == {}
        assert isinstance(sr.generated_at, datetime)


# ---------------------------------------------------------------------------
# ProseSynthesisPipeline
# ---------------------------------------------------------------------------

class TestProseSynthesisPipeline:

    def test_default_themes(self):
        pipeline = ProseSynthesisPipeline()
        assert pipeline.themes == DEFAULT_THEMES
        assert pipeline.min_quality == 0.7
        assert pipeline.passages == []

    def test_custom_themes(self):
        custom = {
            "my_theme": ThemeConfig(
                id="my_theme", name="My Theme", short_name="MT",
                narrative_position=1, keywords=["custom"],
            )
        }
        pipeline = ProseSynthesisPipeline(themes=custom)
        assert "my_theme" in pipeline.themes

    def test_detect_theme(self):
        pipeline = ProseSynthesisPipeline()
        # Text about AI alignment
        theme = pipeline._detect_theme(
            "Artificial intelligence alignment is the key challenge for AI safety"
        )
        assert theme == "ai_risk_evolution"

    def test_detect_theme_systems(self):
        pipeline = ProseSynthesisPipeline()
        theme = pipeline._detect_theme(
            "Complex system dynamics and equilibrium stability patterns emerge from feedback loops"
        )
        assert theme == "systems_complexity"

    def test_load_quality_prose(self, tmp_path):
        data = {
            "passages": [
                {
                    "text": "Intelligence is pattern rather than substance and this has deep implications for understanding consciousness and substrate independence.",
                    "role": "user",
                    "conversation_title": "AI Discussion",
                    "quality_scores": {"overall": 0.9, "depth": 0.8, "novelty": 0.7, "beauty": 0.6},
                },
                {
                    "text": "def function(): pass",
                    "role": "assistant",
                    "conversation_title": "Code Help",
                    "quality_scores": {"overall": 0.9, "depth": 0.5, "novelty": 0.3, "beauty": 0.2},
                },
                {
                    "text": "Low quality passage here.",
                    "role": "user",
                    "conversation_title": "Chat",
                    "quality_scores": {"overall": 0.3, "depth": 0.2, "novelty": 0.1, "beauty": 0.1},
                },
            ]
        }
        f = tmp_path / "prose.json"
        f.write_text(json.dumps(data))

        pipeline = ProseSynthesisPipeline()
        count = pipeline.load_quality_prose(f)

        # Only the first passage should survive (code filtered, low quality filtered)
        assert count == 1
        assert pipeline.passages[0].quality_score == 0.9

    def test_organize_by_theme(self):
        pipeline = ProseSynthesisPipeline()
        pipeline.passages = [
            ProsePassage(
                text="AI risk", role="user", conversation_title="Chat",
                quality_score=0.9, primary_theme="ai_risk_evolution", word_count=2,
            ),
            ProsePassage(
                text="Systems", role="user", conversation_title="Chat",
                quality_score=0.8, primary_theme="systems_complexity", word_count=1,
            ),
        ]
        pipeline._organize_by_theme()
        assert len(pipeline.by_theme["ai_risk_evolution"]) == 1
        assert len(pipeline.by_theme["systems_complexity"]) == 1

    def test_get_statistics(self):
        pipeline = ProseSynthesisPipeline()
        pipeline.passages = [
            ProsePassage(
                text="Text", role="user", conversation_title="Chat",
                quality_score=0.9, primary_theme="ai_risk_evolution", word_count=10,
            ),
        ]
        pipeline.by_theme = {"ai_risk_evolution": pipeline.passages}
        stats = pipeline.get_statistics()
        assert stats["total_passages"] == 1
        assert stats["total_words"] == 10
        assert stats["user_passages"] == 1
        assert stats["avg_quality"] == 0.9

    def test_get_statistics_empty(self):
        pipeline = ProseSynthesisPipeline()
        pipeline.by_theme = {}
        stats = pipeline.get_statistics()
        assert stats["total_passages"] == 0
        assert stats["avg_quality"] == 0

    def test_generate_transition_known(self):
        pipeline = ProseSynthesisPipeline()
        trans = pipeline._generate_transition("ai_risk_evolution", "systems_complexity")
        assert "complex systems" in trans.lower()

    def test_generate_transition_unknown(self):
        pipeline = ProseSynthesisPipeline()
        trans = pipeline._generate_transition("art_aesthetics", "ai_risk_evolution")
        assert "connection" in trans.lower()

    def test_synthesize_unified_essay_no_passages(self):
        pipeline = ProseSynthesisPipeline()
        pipeline.by_theme = {k: [] for k in DEFAULT_THEMES}
        result = pipeline.synthesize_unified_essay()
        assert isinstance(result, SynthesisResult)
        assert result.passages_used == 0

    def test_synthesize_unified_essay_with_passages(self):
        pipeline = ProseSynthesisPipeline()
        passage = ProsePassage(
            text="Intelligence is pattern rather than substance.",
            role="user",
            conversation_title="AI Chat",
            quality_score=0.9,
            primary_theme="intelligence_consciousness",
            word_count=6,
        )
        pipeline.passages = [passage]
        pipeline.by_theme = {k: [] for k in DEFAULT_THEMES}
        pipeline.by_theme["intelligence_consciousness"] = [passage]
        result = pipeline.synthesize_unified_essay(title="Test Essay")
        assert result.title == "Test Essay"
        assert result.passages_used == 1
        assert "intelligence_consciousness" in result.themes_included
        assert "Intelligence is pattern" in result.content

    def test_synthesize_unified_essay_with_thesis(self):
        pipeline = ProseSynthesisPipeline()
        pipeline.by_theme = {k: [] for k in DEFAULT_THEMES}
        result = pipeline.synthesize_unified_essay(thesis="My custom thesis.")
        assert "My custom thesis." in result.content

    def test_synthesize_anthology(self):
        pipeline = ProseSynthesisPipeline()
        passage = ProsePassage(
            text="Test prose passage.", role="user",
            conversation_title="Chat", quality_score=0.9,
            primary_theme="art_aesthetics", word_count=3,
        )
        pipeline.by_theme = {k: [] for k in DEFAULT_THEMES}
        pipeline.by_theme["art_aesthetics"] = [passage]
        results = pipeline.synthesize_anthology()
        assert "art_aesthetics" in results
        assert results["art_aesthetics"].passages_used == 1

    def test_generate_thread_skeleton(self):
        pipeline = ProseSynthesisPipeline()
        pipeline.passages = [
            ProsePassage(
                text="This is a substantial sentence that should be tweetable and interesting enough.",
                role="user", conversation_title="Chat", quality_score=0.9,
                primary_theme="ai_risk_evolution", word_count=12,
            ),
        ]
        thread = pipeline.generate_thread_skeleton(max_tweets=5)
        assert "Thread" in thread or "1." in thread

    def test_export_all(self, tmp_path):
        pipeline = ProseSynthesisPipeline()
        pipeline.passages = []
        pipeline.by_theme = {k: [] for k in DEFAULT_THEMES}
        outputs = pipeline.export_all(tmp_path / "output")
        assert "unified_essay" in outputs
        assert "thread_skeleton" in outputs
        assert "stats" in outputs
        assert (tmp_path / "output" / "unified_essay.md").exists()


class TestCreateProsePipeline:

    def test_creates_pipeline(self):
        pipeline = create_prose_pipeline()
        assert isinstance(pipeline, ProseSynthesisPipeline)
        assert pipeline.min_quality == 0.7

    def test_custom_min_quality(self):
        pipeline = create_prose_pipeline(min_quality=0.5)
        assert pipeline.min_quality == 0.5
