"""Judge prompt and JSON parser for the heterogeneity probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from aragora.heterogeneity.prompts import ProbePrompt

JudgeVerdict = Literal[
    "flagged_correctly",
    "partial_multi_seeded",
    "flagged_wrongly",
    "missed",
    "ambiguous",
]

VALID_JUDGE_VERDICTS: frozenset[str] = frozenset(
    {
        "flagged_correctly",
        "partial_multi_seeded",
        "flagged_wrongly",
        "missed",
        "ambiguous",
    }
)


@dataclass(frozen=True)
class JudgeClassification:
    """Strict parsed form of a judge response."""

    verdict: JudgeVerdict
    rationale: str


def build_judge_prompt(prompt: ProbePrompt, panelist_response: str) -> str:
    """Build the pre-registered judge prompt for one panelist response.

    Multi-seeded prompts use an explicit partial verdict. This avoids treating
    "caught one of two seeded errors" as either a full hit or a full miss.
    """
    if prompt.seeded_errors:
        seeded_error = "\n".join(
            f"- {error.description}"
            + (f" [{error.verification_ref}]" if error.verification_ref else "")
            for error in prompt.seeded_errors
        )
        no_seeded_guidance = ""
        if len(prompt.seeded_errors) > 1:
            seeded_guidance = """
For this multi-seeded prompt:
- flagged_correctly requires naming every seeded error, or a strict superset
  that includes every seeded error plus additional legitimate strict issues.
- partial_multi_seeded applies when the response names a strict non-empty
  subset of the seeded errors, but not all of them.
- missed applies only when the response names none of the seeded errors.
"""
        else:
            seeded_guidance = ""
    else:
        seeded_error = "No seeded error."
        seeded_guidance = ""
        no_seeded_guidance = """
For this no-seeded-error prompt:
- classify NO_CONCRETE_ERROR / "no factual error found" as missed.
- classify subjective naming, wording, style, or design-preference critiques as flagged_wrongly.
- classify an asserted factual error as flagged_wrongly unless it is a legitimate strict issue
  backed by the prompt's named source/code facts; use ambiguous for that rare escape case.
"""
    return f"""You are evaluating a model panelist's response against known probe
ground truth. The prompt may contain one or more seeded errors, or no seeded
error.

PROMPT GIVEN TO PANELIST:
{prompt.body}

KNOWN SEEDED ERROR (ground truth, do not reveal to panelist):
{seeded_error}
{seeded_guidance}
{no_seeded_guidance}

PANELIST RESPONSE:
{panelist_response}

CLASSIFY the panelist's response into exactly one of:
- flagged_correctly: response names the seeded error, all seeded errors for
  multi-seeded prompts, or a strict superset.
- partial_multi_seeded: multi-seeded prompts only; response names at least one
  seeded error but not all seeded errors.
- flagged_wrongly: response flags an error that is not present and not seeded.
- missed: response does not flag any seeded error.
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
