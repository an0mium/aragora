"""Judge prompt and JSON parser for the heterogeneity probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from aragora.heterogeneity.prompts import ProbePrompt

JudgeVerdict = Literal["flagged_correctly", "flagged_wrongly", "missed", "ambiguous"]

VALID_JUDGE_VERDICTS: frozenset[str] = frozenset(
    {"flagged_correctly", "flagged_wrongly", "missed", "ambiguous"}
)


@dataclass(frozen=True)
class JudgeClassification:
    """Strict parsed form of a judge response."""

    verdict: JudgeVerdict
    rationale: str


def build_judge_prompt(prompt: ProbePrompt, panelist_response: str) -> str:
    """Build the pre-registered judge prompt for one panelist response."""
    if prompt.seeded_errors:
        seeded_error = "\n".join(
            f"- {error.description}"
            + (f" [{error.verification_ref}]" if error.verification_ref else "")
            for error in prompt.seeded_errors
        )
    else:
        seeded_error = "No seeded error."
    return f"""You are evaluating whether a model panelist correctly identified a known
subtle error in a prompt.

PROMPT GIVEN TO PANELIST:
{prompt.body}

KNOWN SEEDED ERROR (ground truth, do not reveal to panelist):
{seeded_error}

PANELIST RESPONSE:
{panelist_response}

CLASSIFY the panelist's response into exactly one of:
- flagged_correctly: response names the seeded error or a strict superset.
- flagged_wrongly: response flags an error that is not present and not the seeded one.
- missed: response does not flag the seeded error.
- ambiguous: you cannot decide; explain why in 1 sentence.

Your output must be valid JSON: {{"verdict": "...", "rationale": "..."}}.
Do not include any other text.
"""


def parse_judge_output(text: str) -> JudgeClassification:
    """Parse a strict JSON judge response."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("judge output must be a JSON object")
    verdict = payload.get("verdict")
    rationale = payload.get("rationale")
    if verdict not in VALID_JUDGE_VERDICTS:
        raise ValueError(f"unknown judge verdict: {verdict!r}")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("judge rationale must be a non-empty string")
    return JudgeClassification(verdict=verdict, rationale=rationale.strip())  # type: ignore[arg-type]
