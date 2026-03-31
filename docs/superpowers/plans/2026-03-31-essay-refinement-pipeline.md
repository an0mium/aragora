# Essay Refinement Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `aragora essay refine` — a CLI command that transforms raw ideas into adversarially tested, publishable essays using Aragora's existing multi-model debate infrastructure.

**Architecture:** Six new files under `aragora/essay/` that compose existing debate, workflow, and agent primitives. The pipeline runs: idea extraction (debate) → parallel drafting (multi-agent) → evaluation + critique (rubric scoring) → synthesis (semantic merge) → refinement loop (until score > 0.8 or max rounds) → polish → receipt. The CLI entry point (`aragora essay refine`) orchestrates the pipeline; `aragora essay score` scores an existing draft standalone.

**Tech Stack:** Python 3.11+, existing Aragora debate engine (`Arena`, `DebateProtocol`), `create_agent` factory, `WorkflowEngine` with `LoopStep`, YAML rubric configs, `dataclasses` for `EssayScore`.

**Spec:** `docs/specs/ESSAY_REFINEMENT_PIPELINE.md`

---

## File Structure

```
aragora/essay/                  # NEW directory
├── __init__.py                 # Package exports
├── rubric.py                   # EssayScore dataclass + evaluate_essay() + parse rubric YAML
├── roles.py                    # ESSAY_ROUND_PHASES + role constants
├── prompts.py                  # All prompt templates (extraction, drafting, critique, synthesis, polish)
├── synthesizer.py              # EssaySynthesizer — semantic merge of drafts
├── pipeline.py                 # EssayRefinementPipeline — orchestrates full workflow
└── cli.py                      # CLI handler registered as `aragora essay`

aragora/essay/rubrics/          # NEW directory
├── default.yaml                # Default balanced rubric
└── substack.yaml               # Substack-optimized (shorter, punchier)

aragora/cli/commands/essay.py   # NEW — CLI entry point (thin wrapper calling pipeline.py)
aragora/cli/parser.py           # MODIFY — add essay subcommand

tests/essay/                    # NEW directory
├── __init__.py
├── test_rubric.py
├── test_prompts.py
├── test_synthesizer.py
├── test_pipeline.py
└── test_cli.py
```

---

### Task 1: EssayScore dataclass and rubric parsing

**Files:**
- Create: `aragora/essay/__init__.py`
- Create: `aragora/essay/rubric.py`
- Create: `aragora/essay/rubrics/default.yaml`
- Create: `aragora/essay/rubrics/substack.yaml`
- Create: `tests/essay/__init__.py`
- Create: `tests/essay/test_rubric.py`

- [ ] **Step 1: Write failing tests for EssayScore**

```python
# tests/essay/test_rubric.py
"""Tests for essay evaluation rubric."""
from aragora.essay.rubric import EssayScore, load_rubric, parse_score_response


def test_essay_score_overall_is_weighted_composite():
    score = EssayScore(
        thesis_clarity=0.9,
        argument_coherence=0.8,
        evidence_grounding=0.7,
        rhetorical_force=0.6,
        concision=0.8,
        factual_accuracy=0.9,
        originality=0.5,
    )
    assert 0.0 < score.overall < 1.0
    assert score.overall == score.compute_overall()


def test_essay_score_defaults():
    score = EssayScore()
    assert score.overall == 0.0
    assert score.severity_notes == []
    assert score.suggestions == []


def test_parse_score_response_extracts_json():
    response = '''Some text before
    {"thesis_clarity": 0.9, "argument_coherence": 0.8, "evidence_grounding": 0.7,
     "rhetorical_force": 0.6, "concision": 0.8, "factual_accuracy": 0.9,
     "originality": 0.5, "severity_notes": ["weak opening"], "suggestions": ["add data"]}
    Some text after'''
    score = parse_score_response(response)
    assert score.thesis_clarity == 0.9
    assert score.severity_notes == ["weak opening"]


def test_parse_score_response_handles_missing_fields():
    response = '{"thesis_clarity": 0.5}'
    score = parse_score_response(response)
    assert score.thesis_clarity == 0.5
    assert score.argument_coherence == 0.0  # default


def test_load_rubric_from_yaml(tmp_path):
    rubric_file = tmp_path / "test.yaml"
    rubric_file.write_text("""
name: Test Rubric
weights:
  thesis_clarity: 0.2
  argument_coherence: 0.2
  evidence_grounding: 0.15
  rhetorical_force: 0.15
  concision: 0.1
  factual_accuracy: 0.1
  originality: 0.1
""")
    rubric = load_rubric(str(rubric_file))
    assert rubric["name"] == "Test Rubric"
    assert abs(sum(rubric["weights"].values()) - 1.0) < 0.01


def test_load_default_rubric():
    rubric = load_rubric()  # No path = default
    assert "weights" in rubric
    assert "name" in rubric
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/essay/test_rubric.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aragora.essay'`

- [ ] **Step 3: Implement EssayScore and rubric loading**

```python
# aragora/essay/__init__.py
"""Essay refinement pipeline for Aragora."""
from aragora.essay.rubric import EssayScore, evaluate_essay, load_rubric, parse_score_response

__all__ = ["EssayScore", "evaluate_essay", "load_rubric", "parse_score_response"]
```

```python
# aragora/essay/rubric.py
"""Essay evaluation rubric — LLM-as-judge scoring for prose."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = {
    "thesis_clarity": 0.20,
    "argument_coherence": 0.20,
    "evidence_grounding": 0.15,
    "rhetorical_force": 0.15,
    "concision": 0.10,
    "factual_accuracy": 0.10,
    "originality": 0.10,
}

_RUBRICS_DIR = Path(__file__).parent / "rubrics"


@dataclass
class EssayScore:
    """Structured evaluation of an essay draft."""

    thesis_clarity: float = 0.0
    argument_coherence: float = 0.0
    evidence_grounding: float = 0.0
    rhetorical_force: float = 0.0
    concision: float = 0.0
    factual_accuracy: float = 0.0
    originality: float = 0.0
    overall: float = 0.0

    severity_notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    weakest_paragraph: str = ""
    strongest_paragraph: str = ""
    factual_claims_to_verify: list[str] = field(default_factory=list)

    def compute_overall(self, weights: dict[str, float] | None = None) -> float:
        w = weights or _DEFAULT_WEIGHTS
        self.overall = sum(
            getattr(self, dim, 0.0) * weight
            for dim, weight in w.items()
        )
        return self.overall

    def __post_init__(self) -> None:
        if self.overall == 0.0 and any(
            getattr(self, dim, 0.0) > 0 for dim in _DEFAULT_WEIGHTS
        ):
            self.compute_overall()

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis_clarity": self.thesis_clarity,
            "argument_coherence": self.argument_coherence,
            "evidence_grounding": self.evidence_grounding,
            "rhetorical_force": self.rhetorical_force,
            "concision": self.concision,
            "factual_accuracy": self.factual_accuracy,
            "originality": self.originality,
            "overall": self.overall,
            "severity_notes": self.severity_notes,
            "suggestions": self.suggestions,
        }


def parse_score_response(text: str) -> EssayScore:
    """Parse an LLM response containing JSON scores into an EssayScore."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return EssayScore()
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return EssayScore()

    return EssayScore(
        thesis_clarity=float(data.get("thesis_clarity", 0.0)),
        argument_coherence=float(data.get("argument_coherence", 0.0)),
        evidence_grounding=float(data.get("evidence_grounding", 0.0)),
        rhetorical_force=float(data.get("rhetorical_force", 0.0)),
        concision=float(data.get("concision", 0.0)),
        factual_accuracy=float(data.get("factual_accuracy", 0.0)),
        originality=float(data.get("originality", 0.0)),
        severity_notes=list(data.get("severity_notes", [])),
        suggestions=list(data.get("suggestions", [])),
        weakest_paragraph=str(data.get("weakest_paragraph", "")),
        strongest_paragraph=str(data.get("strongest_paragraph", "")),
        factual_claims_to_verify=list(data.get("factual_claims_to_verify", [])),
    )


def load_rubric(path: str | None = None) -> dict[str, Any]:
    """Load a rubric from YAML. Defaults to built-in default.yaml."""
    import yaml

    target = Path(path) if path else _RUBRICS_DIR / "default.yaml"
    with open(target) as f:
        return yaml.safe_load(f)


async def evaluate_essay(
    essay_text: str,
    judge_agent: Any,
    *,
    rubric: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> EssayScore:
    """Ask a judge agent to score an essay against the rubric."""
    from aragora.essay.prompts import build_evaluation_prompt

    rubric_data = rubric or load_rubric()
    prompt = build_evaluation_prompt(essay_text, rubric_data, context)
    response = await judge_agent.generate(prompt)
    score = parse_score_response(str(getattr(response, "text", response)))
    weights = rubric_data.get("weights") if rubric_data else None
    score.compute_overall(weights)
    return score
```

```yaml
# aragora/essay/rubrics/default.yaml
name: Default Essay Rubric
description: Balanced rubric for general-purpose essay evaluation
weights:
  thesis_clarity: 0.20
  argument_coherence: 0.20
  evidence_grounding: 0.15
  rhetorical_force: 0.15
  concision: 0.10
  factual_accuracy: 0.10
  originality: 0.10
quality_threshold: 0.8
```

```yaml
# aragora/essay/rubrics/substack.yaml
name: Substack Essay Rubric
description: Optimized for conversational, punchy Substack posts
weights:
  thesis_clarity: 0.15
  argument_coherence: 0.15
  evidence_grounding: 0.10
  rhetorical_force: 0.25
  concision: 0.15
  factual_accuracy: 0.10
  originality: 0.10
quality_threshold: 0.75
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/essay/test_rubric.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add aragora/essay/ tests/essay/
git commit -m "feat(essay): add EssayScore rubric and YAML loading"
```

---

### Task 2: Prompt templates

**Files:**
- Create: `aragora/essay/prompts.py`
- Create: `tests/essay/test_prompts.py`

- [ ] **Step 1: Write failing tests for prompt builders**

```python
# tests/essay/test_prompts.py
"""Tests for essay prompt templates."""
from aragora.essay.prompts import (
    build_extraction_prompt,
    build_drafting_prompt,
    build_evaluation_prompt,
    build_synthesis_prompt,
    build_polish_prompt,
)


def test_extraction_prompt_includes_raw_ideas():
    prompt = build_extraction_prompt("My raw idea about AI safety")
    assert "AI safety" in prompt
    assert "thesis" in prompt.lower()
    assert "outline" in prompt.lower()


def test_drafting_prompt_includes_thesis_and_target():
    prompt = build_drafting_prompt(
        thesis="AI will transform education",
        outline="1. Current state\n2. Changes\n3. Implications",
        target_words=1200,
        voice_notes="conversational Substack tone",
    )
    assert "AI will transform education" in prompt
    assert "1200" in prompt
    assert "Substack" in prompt


def test_evaluation_prompt_includes_essay_and_rubric():
    rubric = {"name": "Default", "weights": {"thesis_clarity": 0.2}}
    prompt = build_evaluation_prompt("Essay text here", rubric)
    assert "Essay text here" in prompt
    assert "thesis_clarity" in prompt
    assert "JSON" in prompt


def test_synthesis_prompt_includes_ranked_drafts():
    drafts = [("Draft A text", 0.8), ("Draft B text", 0.6)]
    critiques = ["Draft A has weak opening", "Draft B has strong data"]
    prompt = build_synthesis_prompt(
        ranked_drafts=drafts,
        critiques=critiques,
        target_word_count=1000,
    )
    assert "Draft A" in prompt or "Draft 1" in prompt
    assert "0.8" in prompt or "80" in prompt


def test_polish_prompt_includes_draft_and_word_target():
    prompt = build_polish_prompt("Final draft text", target_words=1200)
    assert "Final draft text" in prompt
    assert "1200" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/essay/test_prompts.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement prompt templates**

```python
# aragora/essay/prompts.py
"""Prompt templates for essay refinement pipeline."""
from __future__ import annotations

from typing import Any


def build_extraction_prompt(
    raw_ideas: str,
    *,
    target_words: int = 1200,
) -> str:
    return f"""You are analyzing a cluster of raw ideas to extract the strongest essay thesis.

## Raw Ideas
{raw_ideas}

## Your Task
1. Identify the 2-3 strongest core claims or insights
2. Identify tensions or contradictions between ideas
3. Propose a single thesis statement that captures the throughline
4. Create a 4-6 section outline for a {target_words}-word essay

## Output Format
Return your analysis as:
- **Thesis:** [one sentence]
- **Outline:**
  1. [section title] — [what this section covers]
  2. ...
- **Key tensions to address:** [list]
- **Strongest supporting evidence:** [from the raw ideas]"""


def build_drafting_prompt(
    thesis: str,
    outline: str,
    *,
    target_words: int = 1200,
    voice_notes: str = "",
) -> str:
    voice = f"\n\n**Voice/Style:** {voice_notes}" if voice_notes else ""
    return f"""Write a complete {target_words}-word essay based on this thesis and outline.

**Thesis:** {thesis}

**Outline:**
{outline}
{voice}

## Rules
- Open with a hook that earns the reader's attention in the first sentence
- Every paragraph must advance the argument — no filler
- Use specific examples, names, dates, or data where possible
- Close with an image or insight that lands, not a summary
- Do NOT use bullet points or headers in the essay body
- Do NOT start with "In today's world" or any generic opener
- Target exactly {target_words} words (±10%)"""


def build_evaluation_prompt(
    essay_text: str,
    rubric: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> str:
    dimensions = "\n".join(
        f"- **{dim}** (weight {w:.0%}): Score 0.0-1.0"
        for dim, w in rubric.get("weights", {}).items()
    )
    ctx = ""
    if context:
        ctx = f"\n\n## Context\n{context.get('notes', '')}"

    return f"""Score this essay against the following rubric. Be rigorous — most essays should score 0.4-0.7 on first draft.

## Rubric: {rubric.get('name', 'Default')}
{dimensions}

## Essay
{essay_text}
{ctx}

## Instructions
1. Score each dimension 0.0-1.0 with a one-sentence justification
2. Identify the single weakest paragraph and explain why
3. Identify the single strongest paragraph and explain why
4. Flag any factual claims that need verification
5. List 3 specific, actionable suggestions for improvement

## Output
Return ONLY a JSON object:
{{"thesis_clarity": 0.X, "argument_coherence": 0.X, "evidence_grounding": 0.X,
  "rhetorical_force": 0.X, "concision": 0.X, "factual_accuracy": 0.X,
  "originality": 0.X, "severity_notes": ["..."], "suggestions": ["..."],
  "weakest_paragraph": "...", "strongest_paragraph": "...",
  "factual_claims_to_verify": ["..."]}}"""


def build_synthesis_prompt(
    ranked_drafts: list[tuple[str, float]],
    critiques: list[str],
    *,
    target_word_count: int = 1200,
    voice_notes: str = "",
) -> str:
    draft_sections = []
    for i, (text, score) in enumerate(ranked_drafts, 1):
        draft_sections.append(f"### Draft {i} (score: {score:.2f})\n{text}")
    drafts_text = "\n\n---\n\n".join(draft_sections)

    critique_text = "\n".join(f"- {c}" for c in critiques)
    voice = f"\n**Voice:** {voice_notes}" if voice_notes else ""

    return f"""You are synthesizing multiple essay drafts into one superior version.

## Drafts (ranked by quality)
{drafts_text}

## Critiques
{critique_text}

## Your Task
1. Take the best structural moves from the highest-scoring draft
2. Incorporate the strongest paragraphs from each draft
3. Address the specific issues raised in the critiques
4. Rewrite the weakest sections
5. Produce a single cohesive {target_word_count}-word essay
{voice}

## Rules
- Do NOT just pick one draft — synthesize across all of them
- Every paragraph must earn its place
- Preserve the best opening and closing from any draft
- Fix factual issues flagged in critiques"""


def build_polish_prompt(
    draft: str,
    *,
    target_words: int = 1200,
    voice_notes: str = "",
) -> str:
    voice = f"\n**Voice:** {voice_notes}" if voice_notes else ""
    return f"""Final polish pass on this essay. Do not change the argument structure.

## Essay
{draft}

## Your Task
1. Tighten every sentence — cut filler words, strengthen verbs
2. Ensure the opening hooks within the first sentence
3. Ensure the closing lands with an image or insight, not a summary
4. Check tone consistency throughout
5. Adjust to within 10% of {target_words} words
6. Remove anything that sounds like a chatbot or AI-generated text
{voice}

Return the polished essay only — no commentary."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/essay/test_prompts.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add aragora/essay/prompts.py tests/essay/test_prompts.py
git commit -m "feat(essay): add prompt templates for all pipeline phases"
```

---

### Task 3: Essay roles and round phases

**Files:**
- Create: `aragora/essay/roles.py`
- Create: `tests/essay/test_roles.py`

- [ ] **Step 1: Write failing test**

```python
# tests/essay/test_roles.py
from aragora.essay.roles import ESSAY_ROUND_PHASES, ESSAY_AGENT_ROLES


def test_essay_phases_are_ordered():
    for i, phase in enumerate(ESSAY_ROUND_PHASES):
        assert phase.number == i


def test_essay_phases_have_required_fields():
    for phase in ESSAY_ROUND_PHASES:
        assert phase.name
        assert phase.description
        assert phase.focus
        assert phase.cognitive_mode


def test_essay_agent_roles_defined():
    assert "drafter" in ESSAY_AGENT_ROLES
    assert "critic" in ESSAY_AGENT_ROLES
    assert "synthesizer" in ESSAY_AGENT_ROLES
    assert "judge" in ESSAY_AGENT_ROLES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/essay/test_roles.py -v`

- [ ] **Step 3: Implement roles**

```python
# aragora/essay/roles.py
"""Essay-specific agent roles and debate round phases."""
from __future__ import annotations

from aragora.debate.protocol import RoundPhase

ESSAY_ROUND_PHASES: list[RoundPhase] = [
    RoundPhase(number=0, name="Idea Extraction", description="Parse raw ideas into core claims, tensions, and thesis candidates", focus="What are the 2-3 strongest ideas? What's the throughline?", cognitive_mode="Analyst"),
    RoundPhase(number=1, name="Parallel Drafting", description="Each agent independently drafts the essay from the extracted thesis", focus="Structure, voice, opening hook, closing image", cognitive_mode="Writer"),
    RoundPhase(number=2, name="Structural Critique", description="Challenge argument structure, logical gaps, unsupported claims", focus="Does the argument hold? Are there missing steps?", cognitive_mode="Skeptic"),
    RoundPhase(number=3, name="Factual Audit", description="Verify every factual claim. Flag anything unverifiable or wrong.", focus="Named people, dates, products, studies, historical claims", cognitive_mode="Fact-Checker"),
    RoundPhase(number=4, name="Devil's Advocate", description="Argue the strongest counter-position", focus="Steelman the opposition. Where is the essay most vulnerable?", cognitive_mode="Devil's Advocate"),
    RoundPhase(number=5, name="Synthesis", description="Merge strongest elements across drafts", focus="Best structure, best paragraphs, best closing from all versions", cognitive_mode="Synthesizer"),
    RoundPhase(number=6, name="Style Polish", description="Tighten prose. Cut filler. Strengthen verbs.", focus="Every sentence earns its place. Remove chatbot-sounding text.", cognitive_mode="Editor"),
    RoundPhase(number=7, name="Final Judgment", description="Score final draft against rubric. Publish or iterate.", focus="Is this publishable? What score does it get?", cognitive_mode="Judge"),
]

ESSAY_AGENT_ROLES: dict[str, dict[str, str]] = {
    "drafter": {"role": "proposer", "description": "Writes complete essay drafts from thesis + outline"},
    "critic": {"role": "critic", "description": "Evaluates drafts against rubric and identifies weaknesses"},
    "fact_checker": {"role": "critic", "description": "Verifies factual claims and flags unverifiable statements"},
    "devils_advocate": {"role": "critic", "description": "Argues the strongest counter-position"},
    "synthesizer": {"role": "synthesizer", "description": "Merges best elements from multiple drafts"},
    "editor": {"role": "synthesizer", "description": "Style polishing — tightens prose, cuts filler"},
    "judge": {"role": "critic", "description": "Final rubric scoring and publish/iterate decision"},
}
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/essay/test_roles.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add aragora/essay/roles.py tests/essay/test_roles.py
git commit -m "feat(essay): add essay-specific agent roles and round phases"
```

---

### Task 4: Essay synthesizer

**Files:**
- Create: `aragora/essay/synthesizer.py`
- Create: `tests/essay/test_synthesizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/essay/test_synthesizer.py
"""Tests for essay synthesizer."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.essay.synthesizer import EssaySynthesizer
from aragora.essay.rubric import EssayScore


@pytest.mark.asyncio
async def test_synthesizer_calls_agent_with_ranked_drafts():
    agent = AsyncMock()
    agent.generate.return_value = MagicMock(text="Synthesized essay text")

    synth = EssaySynthesizer(agent=agent)
    result = await synth.synthesize(
        drafts=["Draft A", "Draft B"],
        scores=[
            EssayScore(thesis_clarity=0.9, argument_coherence=0.8),
            EssayScore(thesis_clarity=0.6, argument_coherence=0.7),
        ],
        critiques=["A has weak opening", "B has strong data"],
        target_words=1000,
    )

    assert result == "Synthesized essay text"
    agent.generate.assert_called_once()
    call_prompt = agent.generate.call_args[0][0]
    assert "Draft 1" in call_prompt or "Draft A" in call_prompt


@pytest.mark.asyncio
async def test_synthesizer_ranks_by_overall_score():
    agent = AsyncMock()
    agent.generate.return_value = MagicMock(text="Result")

    synth = EssaySynthesizer(agent=agent)
    await synth.synthesize(
        drafts=["Low", "High"],
        scores=[
            EssayScore(thesis_clarity=0.3),
            EssayScore(thesis_clarity=0.9),
        ],
        critiques=[],
        target_words=1000,
    )

    prompt = agent.generate.call_args[0][0]
    # Higher-scored draft should appear first
    high_pos = prompt.find("High") if "High" in prompt else -1
    low_pos = prompt.find("Low") if "Low" in prompt else -1
    if high_pos >= 0 and low_pos >= 0:
        assert high_pos < low_pos
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/essay/test_synthesizer.py -v`

- [ ] **Step 3: Implement synthesizer**

```python
# aragora/essay/synthesizer.py
"""Essay synthesis — semantic merge of multiple drafts."""
from __future__ import annotations

import logging
from typing import Any

from aragora.essay.prompts import build_synthesis_prompt
from aragora.essay.rubric import EssayScore

logger = logging.getLogger(__name__)


class EssaySynthesizer:
    """Merge multiple essay drafts using an LLM synthesizer agent."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def synthesize(
        self,
        drafts: list[str],
        scores: list[EssayScore],
        critiques: list[str],
        *,
        target_words: int = 1200,
        voice_notes: str = "",
    ) -> str:
        ranked = sorted(
            zip(drafts, scores),
            key=lambda x: x[1].overall,
            reverse=True,
        )
        ranked_with_scores = [(text, score.overall) for text, score in ranked]

        prompt = build_synthesis_prompt(
            ranked_drafts=ranked_with_scores,
            critiques=critiques,
            target_word_count=target_words,
            voice_notes=voice_notes,
        )

        result = await self.agent.generate(prompt)
        return str(getattr(result, "text", result))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/essay/test_synthesizer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add aragora/essay/synthesizer.py tests/essay/test_synthesizer.py
git commit -m "feat(essay): add semantic draft synthesizer"
```

---

### Task 5: Pipeline orchestrator

**Files:**
- Create: `aragora/essay/pipeline.py`
- Create: `tests/essay/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/essay/test_pipeline.py
"""Tests for essay refinement pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.essay.pipeline import EssayRefinementPipeline
from aragora.essay.rubric import EssayScore


@pytest.mark.asyncio
async def test_pipeline_dry_run_returns_thesis_and_outline():
    pipeline = EssayRefinementPipeline(models=["anthropic-api"])
    with patch.object(pipeline, "_extract_ideas", return_value={
        "thesis": "AI will transform education",
        "outline": "1. Current state\n2. Changes",
    }):
        result = await pipeline.run("Raw ideas about AI", dry_run=True)

    assert result["thesis"] == "AI will transform education"
    assert "outline" in result
    assert "final_essay" not in result


@pytest.mark.asyncio
async def test_pipeline_full_run_produces_essay_and_score():
    pipeline = EssayRefinementPipeline(models=["anthropic-api"])
    with (
        patch.object(pipeline, "_extract_ideas", return_value={
            "thesis": "Test thesis",
            "outline": "1. Section",
        }),
        patch.object(pipeline, "_parallel_draft", return_value=["Draft A", "Draft B"]),
        patch.object(pipeline, "_evaluate_drafts", return_value=(
            [EssayScore(thesis_clarity=0.9, overall=0.85)],
            ["Good structure"],
        )),
        patch.object(pipeline, "_synthesize", return_value="Synthesized draft"),
        patch.object(pipeline, "_polish", return_value="Polished essay"),
        patch.object(pipeline, "_final_score", return_value=EssayScore(overall=0.85)),
    ):
        result = await pipeline.run("Raw ideas", dry_run=False, max_rounds=1)

    assert "final_essay" in result
    assert result["final_essay"] == "Polished essay"
    assert result["final_score"].overall == 0.85


def test_pipeline_config_defaults():
    pipeline = EssayRefinementPipeline()
    assert pipeline.target_words == 1200
    assert pipeline.max_rounds == 3
    assert pipeline.quality_threshold == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/essay/test_pipeline.py -v`

- [ ] **Step 3: Implement pipeline**

```python
# aragora/essay/pipeline.py
"""Essay refinement pipeline — orchestrates the full workflow."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from aragora.essay.rubric import EssayScore, evaluate_essay, load_rubric
from aragora.essay.synthesizer import EssaySynthesizer

logger = logging.getLogger(__name__)


@dataclass
class EssayRefinementPipeline:
    """Orchestrate the essay refinement workflow."""

    models: list[str] = field(default_factory=lambda: ["anthropic-api", "openai-api", "gemini"])
    target_words: int = 1200
    max_rounds: int = 3
    quality_threshold: float = 0.8
    voice_notes: str = ""
    rubric_path: str | None = None

    async def run(
        self,
        raw_ideas: str,
        *,
        dry_run: bool = False,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        rounds = max_rounds or self.max_rounds
        rubric = load_rubric(self.rubric_path)

        # Phase 0: Extract ideas
        extraction = await self._extract_ideas(raw_ideas)
        if dry_run:
            return extraction

        # Phase 1: Parallel drafting
        drafts = await self._parallel_draft(
            extraction["thesis"],
            extraction["outline"],
        )

        # Phase 2-4: Evaluate, critique, synthesize loop
        current_draft = None
        best_score = EssayScore()
        critique_history: list[str] = []

        for round_num in range(rounds):
            logger.info("Essay refinement round %d/%d", round_num + 1, rounds)

            if current_draft is None:
                # First round: evaluate all drafts
                scores, critiques = await self._evaluate_drafts(drafts, rubric)
                critique_history.extend(critiques)
                current_draft = await self._synthesize(drafts, scores, critiques)
            else:
                # Subsequent rounds: evaluate the merged draft
                scores, critiques = await self._evaluate_drafts(
                    [current_draft], rubric
                )
                critique_history.extend(critiques)
                if scores and scores[0].overall >= self.quality_threshold:
                    logger.info(
                        "Quality threshold met: %.2f >= %.2f",
                        scores[0].overall,
                        self.quality_threshold,
                    )
                    break
                current_draft = await self._synthesize(
                    [current_draft], scores, critiques
                )

        # Phase 5: Polish
        final_essay = await self._polish(current_draft or drafts[0])

        # Final score
        final_score = await self._final_score(final_essay, rubric)

        return {
            "final_essay": final_essay,
            "final_score": final_score,
            "thesis": extraction["thesis"],
            "outline": extraction["outline"],
            "rounds_used": round_num + 1,
            "critique_history": critique_history,
        }

    async def _extract_ideas(self, raw_ideas: str) -> dict[str, Any]:
        from aragora.agents.base import create_agent
        from aragora.essay.prompts import build_extraction_prompt

        agent = create_agent(self.models[0], name="extractor", role="proposer")
        prompt = build_extraction_prompt(raw_ideas, target_words=self.target_words)
        response = await agent.generate(prompt)
        text = str(getattr(response, "text", response))

        # Parse thesis and outline from response
        thesis = ""
        outline = ""
        for line in text.split("\n"):
            if line.strip().startswith("**Thesis:**"):
                thesis = line.split("**Thesis:**", 1)[1].strip()
            elif thesis and not outline:
                outline += line + "\n"

        return {
            "thesis": thesis or text[:200],
            "outline": outline.strip() or text,
            "raw_extraction": text,
        }

    async def _parallel_draft(self, thesis: str, outline: str) -> list[str]:
        from aragora.agents.base import create_agent
        from aragora.essay.prompts import build_drafting_prompt

        prompt = build_drafting_prompt(
            thesis, outline,
            target_words=self.target_words,
            voice_notes=self.voice_notes,
        )

        async def draft_with_model(model: str, idx: int) -> str:
            agent = create_agent(model, name=f"drafter-{idx}", role="proposer")
            result = await agent.generate(prompt)
            return str(getattr(result, "text", result))

        tasks = [
            draft_with_model(model, i)
            for i, model in enumerate(self.models[:3])
        ]
        return await asyncio.gather(*tasks)

    async def _evaluate_drafts(
        self,
        drafts: list[str],
        rubric: dict[str, Any],
    ) -> tuple[list[EssayScore], list[str]]:
        from aragora.agents.base import create_agent

        judge = create_agent(self.models[0], name="judge", role="critic")
        scores = []
        critiques = []
        for draft in drafts:
            score = await evaluate_essay(draft, judge, rubric=rubric)
            scores.append(score)
            critiques.extend(score.severity_notes)
            critiques.extend(score.suggestions)
        return scores, critiques

    async def _synthesize(
        self,
        drafts: list[str],
        scores: list[EssayScore],
        critiques: list[str],
    ) -> str:
        from aragora.agents.base import create_agent

        agent = create_agent(self.models[0], name="synthesizer", role="synthesizer")
        synth = EssaySynthesizer(agent=agent)
        return await synth.synthesize(
            drafts, scores, critiques,
            target_words=self.target_words,
            voice_notes=self.voice_notes,
        )

    async def _polish(self, draft: str) -> str:
        from aragora.agents.base import create_agent
        from aragora.essay.prompts import build_polish_prompt

        agent = create_agent(self.models[0], name="editor", role="synthesizer")
        prompt = build_polish_prompt(
            draft, target_words=self.target_words, voice_notes=self.voice_notes
        )
        result = await agent.generate(prompt)
        return str(getattr(result, "text", result))

    async def _final_score(
        self, essay: str, rubric: dict[str, Any]
    ) -> EssayScore:
        from aragora.agents.base import create_agent

        # Use a different model from synthesizer to avoid self-evaluation bias
        judge_model = self.models[1] if len(self.models) > 1 else self.models[0]
        judge = create_agent(judge_model, name="final-judge", role="critic")
        return await evaluate_essay(essay, judge, rubric=rubric)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/essay/test_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add aragora/essay/pipeline.py tests/essay/test_pipeline.py
git commit -m "feat(essay): add pipeline orchestrator with refinement loop"
```

---

### Task 6: CLI command

**Files:**
- Create: `aragora/cli/commands/essay.py`
- Modify: `aragora/cli/parser.py`
- Create: `tests/essay/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

```python
# tests/essay/test_cli.py
"""Tests for essay CLI command."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from argparse import Namespace

from aragora.cli.commands.essay import essay_command
from aragora.essay.rubric import EssayScore


def test_essay_command_dry_run(tmp_path):
    ideas_file = tmp_path / "ideas.md"
    ideas_file.write_text("My ideas about AI safety and alignment")

    args = Namespace(
        subcommand="refine",
        input=str(ideas_file),
        dry_run=True,
        output=None,
        rounds=3,
        models="anthropic-api",
        target_words=1200,
        voice_notes="",
        rubric=None,
        resume=None,
    )

    with patch("aragora.cli.commands.essay.EssayRefinementPipeline") as MockPipeline:
        mock_instance = MockPipeline.return_value
        mock_instance.run = AsyncMock(return_value={
            "thesis": "AI alignment requires adversarial testing",
            "outline": "1. Current state\n2. The gap",
        })
        essay_command(args)
        mock_instance.run.assert_called_once()


def test_essay_score_subcommand(tmp_path):
    draft_file = tmp_path / "draft.md"
    draft_file.write_text("This is my essay draft about technology.")

    args = Namespace(
        subcommand="score",
        input=str(draft_file),
        rubric=None,
        models="anthropic-api",
    )

    with patch("aragora.cli.commands.essay.evaluate_essay", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = EssayScore(thesis_clarity=0.8, overall=0.75)
        essay_command(args)
        mock_eval.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/essay/test_cli.py -v`

- [ ] **Step 3: Implement CLI command**

```python
# aragora/cli/commands/essay.py
"""CLI command for essay refinement."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def essay_command(args: Any) -> None:
    """Entry point for `aragora essay` command."""
    subcommand = getattr(args, "subcommand", "refine")

    if subcommand == "score":
        _score_command(args)
    else:
        _refine_command(args)


def _refine_command(args: Any) -> None:
    from aragora.essay.pipeline import EssayRefinementPipeline

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    raw_ideas = input_path.read_text(encoding="utf-8")
    models = [m.strip() for m in (args.models or "anthropic-api").split(",")]

    pipeline = EssayRefinementPipeline(
        models=models,
        target_words=int(args.target_words or 1200),
        max_rounds=int(args.rounds or 3),
        voice_notes=args.voice_notes or "",
        rubric_path=args.rubric,
    )

    result = asyncio.run(pipeline.run(
        raw_ideas,
        dry_run=bool(args.dry_run),
        max_rounds=int(args.rounds or 3),
    ))

    if args.dry_run:
        print(f"\nThesis: {result.get('thesis', '')}")
        print(f"\nOutline:\n{result.get('outline', '')}")
        return

    # Write output
    final_essay = result.get("final_essay", "")
    output_path = args.output
    if output_path:
        Path(output_path).write_text(final_essay, encoding="utf-8")
        print(f"Essay saved to {output_path}")
    else:
        print(final_essay)

    # Print score summary
    score = result.get("final_score")
    if score:
        print(f"\n--- Score: {score.overall:.2f} ---")
        print(f"Rounds used: {result.get('rounds_used', '?')}")
        for dim in ["thesis_clarity", "argument_coherence", "evidence_grounding",
                     "rhetorical_force", "concision", "factual_accuracy", "originality"]:
            print(f"  {dim}: {getattr(score, dim, 0):.2f}")


def _score_command(args: Any) -> None:
    from aragora.agents.base import create_agent
    from aragora.essay.rubric import evaluate_essay, load_rubric

    input_path = Path(args.input)
    essay_text = input_path.read_text(encoding="utf-8")
    models = [m.strip() for m in (args.models or "anthropic-api").split(",")]

    judge = create_agent(models[0], name="judge", role="critic")
    rubric = load_rubric(args.rubric)
    score = asyncio.run(evaluate_essay(essay_text, judge, rubric=rubric))

    print(f"\n--- Essay Score: {score.overall:.2f} ---")
    for dim in ["thesis_clarity", "argument_coherence", "evidence_grounding",
                 "rhetorical_force", "concision", "factual_accuracy", "originality"]:
        print(f"  {dim}: {getattr(score, dim, 0):.2f}")
    if score.severity_notes:
        print(f"\nIssues: {', '.join(score.severity_notes)}")
    if score.suggestions:
        print(f"\nSuggestions:")
        for s in score.suggestions:
            print(f"  - {s}")
```

- [ ] **Step 4: Register in parser.py**

Add to `aragora/cli/parser.py` — find the subcommands section and add:

```python
# In build_parser(), add after other subcommand registrations:
essay_parser = subparsers.add_parser("essay", help="Refine essays through multi-round adversarial debate")
essay_sub = essay_parser.add_subparsers(dest="subcommand")

refine_parser = essay_sub.add_parser("refine", help="Transform raw ideas into a polished essay")
refine_parser.add_argument("--input", "-i", required=True, help="Path to raw ideas file")
refine_parser.add_argument("--output", "-o", help="Output file path")
refine_parser.add_argument("--rounds", "-r", type=int, default=3, help="Max refinement rounds")
refine_parser.add_argument("--models", "-m", default="anthropic-api,openai-api,gemini", help="Comma-separated model list")
refine_parser.add_argument("--target-words", type=int, default=1200, help="Target word count")
refine_parser.add_argument("--voice-notes", default="", help="Style guidance")
refine_parser.add_argument("--rubric", help="Path to custom rubric YAML")
refine_parser.add_argument("--dry-run", action="store_true", help="Extract thesis only")
refine_parser.add_argument("--resume", help="Resume from checkpoint ID")

score_parser = essay_sub.add_parser("score", help="Score an existing draft against rubric")
score_parser.add_argument("--input", "-i", required=True, help="Path to essay file")
score_parser.add_argument("--rubric", help="Path to custom rubric YAML")
score_parser.add_argument("--models", "-m", default="anthropic-api", help="Judge model")

essay_parser.set_defaults(func=_lazy("aragora.cli.commands.essay", "essay_command"))
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/essay/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/essay/ -v`
Expected: All tests pass (16+ tests across all files)

- [ ] **Step 7: Commit**

```bash
git add aragora/cli/commands/essay.py aragora/cli/parser.py tests/essay/test_cli.py
git commit -m "feat(essay): add CLI command for essay refinement and scoring"
```

---

### Task 7: Integration test with mock agents

**Files:**
- Modify: `tests/essay/test_pipeline.py` (add integration test)

- [ ] **Step 1: Write integration test**

```python
# Add to tests/essay/test_pipeline.py

@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_mocked_agents():
    """Full pipeline run with mocked agent responses."""
    with patch("aragora.agents.base.create_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.generate.side_effect = [
            # Extraction response
            MagicMock(text="**Thesis:** AI needs adversarial testing\n**Outline:**\n1. Problem\n2. Solution"),
            # Draft 1
            MagicMock(text="Draft about AI adversarial testing. " * 100),
            # Draft 2
            MagicMock(text="Alternative draft about AI safety. " * 100),
            # Draft 3
            MagicMock(text="Third perspective on AI testing. " * 100),
            # Evaluation (JSON score)
            MagicMock(text='{"thesis_clarity": 0.8, "argument_coherence": 0.7, "evidence_grounding": 0.6, "rhetorical_force": 0.7, "concision": 0.8, "factual_accuracy": 0.9, "originality": 0.5, "severity_notes": ["weak opening"], "suggestions": ["add data"]}'),
            MagicMock(text='{"thesis_clarity": 0.7, "argument_coherence": 0.8, "evidence_grounding": 0.7, "rhetorical_force": 0.6, "concision": 0.7, "factual_accuracy": 0.8, "originality": 0.6}'),
            MagicMock(text='{"thesis_clarity": 0.6, "argument_coherence": 0.6, "evidence_grounding": 0.8, "rhetorical_force": 0.5, "concision": 0.6, "factual_accuracy": 0.7, "originality": 0.7}'),
            # Synthesis
            MagicMock(text="Synthesized essay combining best elements. " * 100),
            # Polish
            MagicMock(text="Polished final essay. " * 100),
            # Final score
            MagicMock(text='{"thesis_clarity": 0.85, "argument_coherence": 0.82, "evidence_grounding": 0.78, "rhetorical_force": 0.80, "concision": 0.85, "factual_accuracy": 0.90, "originality": 0.70}'),
        ]
        mock_create.return_value = mock_agent

        pipeline = EssayRefinementPipeline(
            models=["anthropic-api", "openai-api", "gemini"],
            target_words=1000,
            max_rounds=1,
        )
        result = await pipeline.run("Raw ideas about AI adversarial testing")

        assert "final_essay" in result
        assert "final_score" in result
        assert result["final_score"].overall > 0
        assert result["thesis"]
        assert result["rounds_used"] >= 1
```

- [ ] **Step 2: Run full suite**

Run: `python -m pytest tests/essay/ -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/essay/test_pipeline.py
git commit -m "test(essay): add end-to-end integration test with mocked agents"
```

- [ ] **Step 4: Verify CLI help**

Run: `python -m aragora.cli.main essay --help`
Expected: Shows refine/score subcommands with all flags

- [ ] **Step 5: Final commit with all files**

```bash
git add -A
git commit -m "feat(essay): complete essay refinement pipeline

New CLI commands:
  aragora essay refine --input ideas.md --output essay.md
  aragora essay score --input draft.md

Pipeline: idea extraction → parallel drafting → evaluation → synthesis → refinement loop → polish
Supports custom rubrics (YAML), configurable models, voice/style notes, and dry-run mode."
```
