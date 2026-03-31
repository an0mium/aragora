"""Prompt-builder functions for all Essay Refinement Pipeline phases."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Phase 1 — Extraction
# ---------------------------------------------------------------------------


def build_extraction_prompt(raw_ideas: str, *, target_words: int = 1200) -> str:
    """Return a prompt that asks the model to extract a thesis and outline.

    Parameters
    ----------
    raw_ideas:
        Unstructured notes or brainstorm text.
    target_words:
        Approximate word count for the final essay (used for scope guidance).
    """
    return (
        f"You are a skilled editor helping to structure an essay of approximately "
        f"{target_words} words.\n\n"
        "Given the raw ideas below, extract:\n"
        "1. A clear, arguable **thesis** (one sentence).\n"
        "2. A structured **outline** (numbered sections with brief descriptions).\n\n"
        "Return your response in this format:\n"
        "THESIS: <thesis sentence>\n"
        "OUTLINE:\n"
        "<numbered outline>\n\n"
        "Raw ideas:\n"
        f"{raw_ideas}"
    )


# ---------------------------------------------------------------------------
# Phase 2 — Drafting
# ---------------------------------------------------------------------------


def build_drafting_prompt(
    thesis: str,
    outline: str,
    *,
    target_words: int = 1200,
    voice_notes: str = "",
) -> str:
    """Return a prompt to write a full essay draft from a thesis and outline.

    Parameters
    ----------
    thesis:
        The central arguable claim.
    outline:
        Structured section plan.
    target_words:
        Target essay length in words.
    voice_notes:
        Optional stylistic guidance from the author.
    """
    voice_section = f"\nVoice / style notes: {voice_notes}\n" if voice_notes else ""

    return (
        f"You are an expert essayist. Write a {target_words}-word essay based on the "
        "thesis and outline below.\n\n"
        "Rules — you MUST follow these:\n"
        "• Hook: the first sentence must hook the reader immediately; no filler opening.\n"
        "• No filler: eliminate all throat-clearing phrases (e.g. 'In this essay I will…').\n"
        "• No bullet points: write in flowing prose paragraphs only.\n"
        "• Stay on argument: every sentence must serve the thesis.\n"
        f"{voice_section}\n"
        f"Thesis: {thesis}\n\n"
        f"Outline:\n{outline}\n\n"
        "Write the essay now:"
    )


# ---------------------------------------------------------------------------
# Phase 3 — Evaluation
# ---------------------------------------------------------------------------


def build_evaluation_prompt(
    essay_text: str,
    rubric: dict[str, Any],
    context: str | None = None,
) -> str:
    """Return a prompt that scores an essay against the rubric dimensions.

    The model is instructed to return a JSON object with all dimension scores
    and qualitative feedback fields.

    Parameters
    ----------
    essay_text:
        The essay to be scored.
    rubric:
        Rubric dict (from ``load_rubric``).  Weights are shown to the model.
    context:
        Optional extra context (e.g. topic, audience).
    """
    weights = rubric.get("weights", {})
    dimension_lines = (
        "\n".join(f"- {dim} (weight: {weight:.2f})" for dim, weight in weights.items())
        if weights
        else (
            "- thesis_clarity\n"
            "- argument_coherence\n"
            "- evidence_grounding\n"
            "- rhetorical_force\n"
            "- concision\n"
            "- factual_accuracy\n"
            "- originality"
        )
    )

    context_section = f"\nContext:\n{context}\n" if context else ""

    return (
        "You are an expert essay evaluator. Score the following essay on these "
        "dimensions (0.0–1.0):\n\n"
        f"{dimension_lines}\n\n"
        "Also provide:\n"
        "- severity_notes: list of critical issues\n"
        "- suggestions: list of improvement ideas\n"
        "- weakest_paragraph: identify the weakest paragraph\n"
        "- strongest_paragraph: identify the strongest paragraph\n"
        "- factual_claims_to_verify: list of claims needing fact-checking\n\n"
        "Return ONLY a JSON object with these fields. No prose outside the JSON.\n"
        f"{context_section}\n"
        f"Essay:\n{essay_text}"
    )


# ---------------------------------------------------------------------------
# Phase 4 — Synthesis
# ---------------------------------------------------------------------------


def build_synthesis_prompt(
    ranked_drafts: list[tuple[str, float]],
    critiques: list[str],
    *,
    target_word_count: int = 1200,
    voice_notes: str = "",
) -> str:
    """Return a prompt to merge multiple ranked drafts into a single best essay.

    Parameters
    ----------
    ranked_drafts:
        List of ``(draft_text, score)`` tuples, ordered best-first.
    critiques:
        Critique strings corresponding to each draft.
    target_word_count:
        Desired word count for the synthesised essay.
    voice_notes:
        Optional stylistic guidance.
    """
    draft_sections = []
    for i, (draft_text, score) in enumerate(ranked_drafts, start=1):
        draft_sections.append(f"--- Draft {i} (score: {score:.2f}) ---\n{draft_text}")
    drafts_block = "\n\n".join(draft_sections)

    critiques_block = ""
    if critiques:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(critiques, start=1))
        critiques_block = f"\nCritiques:\n{numbered}\n"

    voice_section = f"\nVoice / style notes: {voice_notes}\n" if voice_notes else ""

    return (
        f"You are a master editor. Synthesise the following ranked drafts into a single "
        f"best-in-class {target_word_count}-word essay.\n\n"
        "Instructions:\n"
        "• Preserve the strongest arguments and phrasing from the highest-scoring drafts.\n"
        "• Address the critiques listed below.\n"
        "• Produce flowing prose — no bullet points.\n"
        f"{voice_section}"
        f"\n{drafts_block}\n"
        f"{critiques_block}\n"
        "Write the synthesised essay now:"
    )


# ---------------------------------------------------------------------------
# Phase 5 — Polish
# ---------------------------------------------------------------------------


def build_polish_prompt(
    draft: str,
    *,
    target_words: int = 1200,
    voice_notes: str = "",
) -> str:
    """Return a prompt for final style polish on a near-final draft.

    Parameters
    ----------
    draft:
        The essay draft to polish.
    target_words:
        Target word count (used for trimming guidance).
    voice_notes:
        Optional stylistic guidance.
    """
    voice_section = f"\nVoice / style notes: {voice_notes}\n" if voice_notes else ""

    return (
        f"You are a copy-editor performing a final polish pass on a {target_words}-word essay.\n\n"
        "Rules — you MUST follow these:\n"
        "• Do not change argument structure — preserve all claims and their order.\n"
        "• Improve sentence variety, word choice, and flow.\n"
        "• Remove redundant phrases and tighten prose.\n"
        "• Fix any grammatical or punctuation errors.\n"
        f"{voice_section}\n"
        f"Draft:\n{draft}\n\n"
        "Return the polished essay only, with no commentary:"
    )
