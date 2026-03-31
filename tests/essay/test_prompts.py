"""Tests for essay prompt-builder functions — written FIRST (TDD)."""

from __future__ import annotations


from aragora.essay.prompts import (
    build_drafting_prompt,
    build_evaluation_prompt,
    build_extraction_prompt,
    build_polish_prompt,
    build_synthesis_prompt,
)


# ---------------------------------------------------------------------------
# 1. build_extraction_prompt
# ---------------------------------------------------------------------------


def test_extraction_prompt_contains_thesis_and_outline():
    """Extraction prompt must ask for thesis and outline, and include the raw ideas."""
    raw_ideas = "AI is changing education; students need critical thinking"
    prompt = build_extraction_prompt(raw_ideas)

    assert isinstance(prompt, str)
    assert raw_ideas in prompt
    assert "thesis" in prompt.lower()
    assert "outline" in prompt.lower()
    assert "1200" in prompt  # default target_words


def test_extraction_prompt_respects_target_words():
    """Custom target_words is reflected in the prompt."""
    prompt = build_extraction_prompt("some ideas", target_words=800)
    assert "800" in prompt


# ---------------------------------------------------------------------------
# 2. build_drafting_prompt
# ---------------------------------------------------------------------------


def test_drafting_prompt_contains_rules_and_content():
    """Drafting prompt must include thesis, no-filler rule, hook rule, no-bullets rule."""
    thesis = "Remote work permanently reshapes urban geography"
    outline = "1. Intro\n2. Evidence\n3. Counter\n4. Conclusion"
    prompt = build_drafting_prompt(thesis, outline)

    assert isinstance(prompt, str)
    assert thesis in prompt
    assert outline in prompt
    # Rules
    assert "filler" in prompt.lower()
    assert "hook" in prompt.lower()
    assert "bullet" in prompt.lower()
    assert "1200" in prompt  # default target_words


def test_drafting_prompt_includes_voice_notes():
    """When voice_notes is provided, it appears in the drafting prompt."""
    prompt = build_drafting_prompt("thesis", "outline", voice_notes="Use conversational tone")
    assert "conversational tone" in prompt


# ---------------------------------------------------------------------------
# 3. build_evaluation_prompt
# ---------------------------------------------------------------------------


def test_evaluation_prompt_lists_dimensions_and_requests_json():
    """Evaluation prompt must list all 7 rubric dimensions and request JSON output."""
    essay = "This is a sample essay about climate change."
    rubric = {
        "name": "default",
        "weights": {
            "thesis_clarity": 0.20,
            "argument_coherence": 0.20,
            "evidence_grounding": 0.15,
            "rhetorical_force": 0.15,
            "concision": 0.10,
            "factual_accuracy": 0.10,
            "originality": 0.10,
        },
    }
    prompt = build_evaluation_prompt(essay, rubric)

    assert isinstance(prompt, str)
    assert essay in prompt
    # All 7 dimensions must be named
    for dim in (
        "thesis_clarity",
        "argument_coherence",
        "evidence_grounding",
        "rhetorical_force",
        "concision",
        "factual_accuracy",
        "originality",
    ):
        assert dim in prompt
    # Must request JSON
    assert "json" in prompt.lower()


def test_evaluation_prompt_includes_context():
    """Optional context parameter is included in the evaluation prompt."""
    prompt = build_evaluation_prompt("essay text", {}, context="Focus on economic arguments")
    assert "economic arguments" in prompt


# ---------------------------------------------------------------------------
# 4. build_synthesis_prompt
# ---------------------------------------------------------------------------


def test_synthesis_prompt_numbers_drafts_with_scores():
    """Synthesis prompt must include numbered drafts with their scores."""
    ranked_drafts = [
        ("Draft text A about AI.", 0.87),
        ("Draft text B about AI.", 0.74),
    ]
    critiques = ["Draft A lacks counter-argument", "Draft B is too verbose"]
    prompt = build_synthesis_prompt(ranked_drafts, critiques)

    assert isinstance(prompt, str)
    # Each draft should appear with its score
    assert "Draft text A about AI." in prompt
    assert "Draft text B about AI." in prompt
    assert "0.87" in prompt
    assert "0.74" in prompt
    # Critiques should appear
    for c in critiques:
        assert c in prompt
    # Target word count
    assert "1200" in prompt


def test_synthesis_prompt_respects_voice_notes():
    """Voice notes appear in the synthesis prompt."""
    prompt = build_synthesis_prompt(
        [("draft", 0.9)],
        [],
        voice_notes="maintain academic register",
    )
    assert "academic register" in prompt


# ---------------------------------------------------------------------------
# 5. build_polish_prompt
# ---------------------------------------------------------------------------


def test_polish_prompt_preserves_argument_structure_instruction():
    """Polish prompt must instruct the model not to change argument structure."""
    draft = "The industrial revolution transformed society in three key ways."
    prompt = build_polish_prompt(draft)

    assert isinstance(prompt, str)
    assert draft in prompt
    assert "argument structure" in prompt.lower()
    assert "do not change" in prompt.lower()
    assert "1200" in prompt  # default target_words


def test_polish_prompt_includes_voice_notes():
    """Voice notes are forwarded into the polish prompt."""
    prompt = build_polish_prompt("my draft", voice_notes="Hemingway style")
    assert "Hemingway style" in prompt
