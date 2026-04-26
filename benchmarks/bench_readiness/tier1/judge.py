"""LLM-judge — Claude Opus 4.7 grades two blinded answers side-by-side.

For ``exact_match`` tasks, the judge is bypassed and we just check the
ground truth. For ``llm_judge`` and ``test_based`` (fallback) tasks we
randomize A/B assignment, ask Opus to score four dimensions, and extract
a winner.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from benchmarks.bench_readiness.tier1.systems.base import SystemOutput
from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

JUDGE_MODEL = "claude-opus-4-7"
JUDGE_SYSTEM = (
    "You are an impartial expert judge evaluating two anonymous answers to "
    "the same question. Score each answer independently, then pick a winner. "
    "You must output ONLY valid JSON matching the schema; no preamble, no "
    "markdown fences, no trailing text."
)

JUDGE_SCHEMA_HINT = """{
  "answer_a": {
    "correctness": 0-10,
    "completeness": 0-10,
    "reasoning_quality": 0-10,
    "usefulness": 0-10
  },
  "answer_b": {
    "correctness": 0-10,
    "completeness": 0-10,
    "reasoning_quality": 0-10,
    "usefulness": 0-10
  },
  "winner": "A" | "B" | "TIE",
  "rationale": "one paragraph explaining the winner"
}"""


@dataclass
class JudgeVerdict:
    """One judge's scoring of two answers for a single task."""

    task_id: str
    a_system: str
    b_system: str
    scores_a: dict[str, int]
    scores_b: dict[str, int]
    winner_system: str  # canonical system name (not "A"/"B")
    rationale: str
    exact_match_used: bool = False
    error: str = ""


def _extract_final_letter(answer: str) -> str | None:
    """For MMLU-Pro: pull the final 'Answer: X' letter out of free text."""
    if not answer:
        return None
    match = re.search(
        r"(?:final\s+)?answer\s*[:\-]\s*\(?([A-J])\)?",
        answer,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Fallback: last isolated capital letter in the last 100 chars
    tail = answer.strip()[-120:]
    letters = re.findall(r"\b([A-J])\b", tail)
    if letters:
        return letters[-1].upper()
    return None


def _judge_exact_match(task: TaskItem, solo: SystemOutput, debate: SystemOutput) -> JudgeVerdict:
    """Score MMLU-Pro style items using ground-truth letter match."""
    ref = (task.reference_answer or "").strip().upper()

    solo_letter = _extract_final_letter(solo.answer) or ""
    debate_letter = _extract_final_letter(debate.answer) or ""

    solo_correct = solo_letter == ref
    debate_correct = debate_letter == ref

    # Translate to 0-10 scale
    def _score(correct: bool) -> dict[str, int]:
        return {
            "correctness": 10 if correct else 0,
            "completeness": 10 if correct else 5,
            "reasoning_quality": 5,  # not scored without LLM-judge
            "usefulness": 10 if correct else 5,
        }

    if solo_correct and not debate_correct:
        winner = solo.system
    elif debate_correct and not solo_correct:
        winner = debate.system
    else:
        winner = "TIE"

    return JudgeVerdict(
        task_id=task.task_id,
        a_system=solo.system,
        b_system=debate.system,
        scores_a=_score(solo_correct),
        scores_b=_score(debate_correct),
        winner_system=winner,
        rationale=(
            f"Reference: {ref}. Solo extracted: {solo_letter or '(none)'}, "
            f"Debate extracted: {debate_letter or '(none)'}."
        ),
        exact_match_used=True,
    )


def _judge_llm(
    task: TaskItem,
    solo: SystemOutput,
    debate: SystemOutput,
    *,
    api_key: str,
    seed: int,
) -> JudgeVerdict:
    """Score a pair of answers using Opus 4.7 as the judge.

    A/B assignment is randomized (with seed) to prevent position bias.
    """
    import anthropic

    rng = random.Random(seed)
    a_is_solo = rng.random() < 0.5
    if a_is_solo:
        a_system, a_answer = solo.system, solo.answer
        b_system, b_answer = debate.system, debate.answer
    else:
        a_system, a_answer = debate.system, debate.answer
        b_system, b_answer = solo.system, solo.answer

    rubric_block = ""
    if task.reference_answer:
        rubric_block = f"\n\nRubric (not visible to the answerers):\n{task.reference_answer}\n"

    user_prompt = (
        f"Question:\n{task.prompt}\n"
        f"{('Context:' + chr(10) + task.context + chr(10) + chr(10)) if task.context else ''}"
        f"{rubric_block}\n"
        f"Answer A:\n{a_answer or '(no answer)'}\n\n"
        f"Answer B:\n{b_answer or '(no answer)'}\n\n"
        f"Output JSON matching this schema exactly:\n{JUDGE_SCHEMA_HINT}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = ""
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                raw_text += block.text

        # Tolerate stray fences or whitespace around the JSON
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)

        scores_a = {
            k: int(parsed["answer_a"].get(k, 0))
            for k in ("correctness", "completeness", "reasoning_quality", "usefulness")
        }
        scores_b = {
            k: int(parsed["answer_b"].get(k, 0))
            for k in ("correctness", "completeness", "reasoning_quality", "usefulness")
        }
        winner_letter = (parsed.get("winner") or "TIE").strip().upper()
        rationale = str(parsed.get("rationale", ""))[:2000]
    except Exception as e:  # noqa: BLE001 - provider SDKs can raise heterogeneous errors
        return JudgeVerdict(
            task_id=task.task_id,
            a_system=a_system,
            b_system=b_system,
            scores_a={},
            scores_b={},
            winner_system="",
            rationale="",
            error=f"judge_error: {type(e).__name__}: {e}",
        )

    if winner_letter == "A":
        winner_system = a_system
    elif winner_letter == "B":
        winner_system = b_system
    else:
        winner_system = "TIE"

    # Re-map scores back to solo vs debate (not A vs B) for CSV clarity
    solo_scores = scores_a if a_is_solo else scores_b
    debate_scores = scores_b if a_is_solo else scores_a

    return JudgeVerdict(
        task_id=task.task_id,
        a_system=solo.system,
        b_system=debate.system,
        scores_a=solo_scores,
        scores_b=debate_scores,
        winner_system=winner_system,
        rationale=rationale,
    )


def judge(
    task: TaskItem,
    solo: SystemOutput,
    debate: SystemOutput,
    *,
    api_key: str,
    seed: int = 0,
) -> JudgeVerdict:
    """Score a single (task, solo_output, debate_output) triple.

    Uses exact-match for ``eval_strategy="exact_match"`` tasks, otherwise
    routes to the LLM judge.
    """
    if solo.error or debate.error:
        return JudgeVerdict(
            task_id=task.task_id,
            a_system=solo.system,
            b_system=debate.system,
            scores_a={},
            scores_b={},
            winner_system="",
            rationale="",
            error=f"system_errors: solo={solo.error!r} debate={debate.error!r}",
        )

    if task.eval_strategy == "exact_match":
        return _judge_exact_match(task, solo, debate)

    return _judge_llm(task, solo, debate, api_key=api_key, seed=seed)
