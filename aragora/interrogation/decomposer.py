"""Decompose vague prompts into concrete dimensions for interrogation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Dimension:
    """A concrete dimension extracted from a vague prompt."""

    name: str
    description: str
    vagueness_score: float  # 0.0 = concrete, 1.0 = very vague
    keywords: list[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Result of decomposing a vague prompt into dimensions."""

    original_prompt: str
    dimensions: list[Dimension]
    overall_vagueness: float  # average vagueness across dimensions

    @property
    def needs_interrogation(self) -> bool:
        return self.overall_vagueness > 0.3 or len(self.dimensions) > 3


# Heuristic patterns for dimension extraction
_DIMENSION_PATTERNS: list[tuple[str, str, float]] = [
    (r"\b(fast|slow|perf|speed|latency)\b", "performance", 0.6),
    (r"\b(ui|ux|design|visual|look|feel)\b", "user-experience", 0.7),
    (r"\b(test|coverage|quality|bug|fix)\b", "quality", 0.4),
    (r"\b(feature|add|new|capability)\b", "functionality", 0.7),
    (r"\b(security|auth|encrypt|safe)\b", "security", 0.5),
    (r"\b(scale|deploy|infra|cloud)\b", "infrastructure", 0.6),
    (r"\b(doc|readme|guide|explain)\b", "documentation", 0.3),
    (r"\b(refactor|clean|organize|simplify)\b", "maintainability", 0.5),
]

_VAGUE_AMPLIFIERS = [
    r"\bmore\b",
    r"\bbetter\b",
    r"\bimprove\b",
    r"\benhance\b",
    r"\bpowerful\b",
    r"\buseful\b",
    r"\bgreat\b",
]


class InterrogationDecomposer:
    """Decomposes vague prompts into concrete dimensions for interrogation."""

    def decompose(self, prompt: str) -> DecompositionResult:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        prompt_lower = prompt.lower()
        dimensions: list[Dimension] = []

        for pattern, name, base_vagueness in _DIMENSION_PATTERNS:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                vagueness = base_vagueness
                for amp in _VAGUE_AMPLIFIERS:
                    if re.search(amp, prompt_lower):
                        vagueness = min(1.0, vagueness + 0.15)
                        break

                keywords = re.findall(pattern, prompt_lower, re.IGNORECASE)
                dimensions.append(
                    Dimension(
                        name=name,
                        description=f"Relates to {name} aspects of the request",
                        vagueness_score=round(vagueness, 2),
                        keywords=keywords,
                    )
                )

        if not dimensions:
            has_amplifier = any(re.search(amp, prompt_lower) for amp in _VAGUE_AMPLIFIERS)
            if has_amplifier:
                # Vague prompt with no specific keywords — surface multiple
                # candidate dimensions so the interrogation engine can ask
                # which ones the user actually cares about.
                for name, base in (
                    ("performance", 0.8),
                    ("functionality", 0.9),
                    ("quality", 0.7),
                ):
                    dimensions.append(
                        Dimension(
                            name=name,
                            description=f"Possible {name} intent in: {prompt[:80]}",
                            vagueness_score=base,
                            keywords=[],
                        )
                    )
            else:
                dimensions.append(
                    Dimension(
                        name="general",
                        description=f"General request: {prompt[:100]}",
                        vagueness_score=0.5,
                        keywords=[],
                    )
                )

        overall = sum(d.vagueness_score for d in dimensions) / len(dimensions)

        return DecompositionResult(
            original_prompt=prompt,
            dimensions=dimensions,
            overall_vagueness=round(overall, 2),
        )
