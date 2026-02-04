"""Lightweight ClaimCheck used by evidence grounding and tests."""

from __future__ import annotations

from dataclasses import dataclass
import re

from aragora.evidence.collector import EvidencePack, EvidenceSnippet


_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class ClaimCheckConfig:
    """Configuration for ClaimCheck heuristics."""

    min_overlap_words: int = 2


@dataclass
class EvidenceMatch:
    """Match between a claim and an evidence snippet."""

    snippet: EvidenceSnippet
    score: float
    overlap_words: int
    number_matches: int


class ClaimCheck:
    """Simple claim checker used for grounding claims to evidence.

    This is a lightweight compatibility layer for tests and EvidenceGrounder.
    It uses token overlap and numeric matches to score evidence snippets.
    """

    def __init__(self, config: ClaimCheckConfig | None = None):
        self.config = config or ClaimCheckConfig()

    def extract_atomic_claims(self, text: str) -> list[str]:
        """Extract atomic claims from a block of text."""
        if not text:
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        bullets: list[str] = []
        for line in lines:
            match = _BULLET_RE.match(line)
            if match:
                bullets.append(match.group(1).strip())

        if bullets:
            return bullets

        # Split on common conjunctions as a simple heuristic.
        parts = re.split(r"\s+and\s+|\s+&\s+", text)
        if len(parts) > 1:
            return [part.strip().strip(".") for part in parts if part.strip()]

        return [text.strip()]

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        return {t for t in tokens if len(t) > 1}

    def _extract_numbers(self, text: str) -> set[str]:
        return set(_NUMBER_RE.findall(text))

    def match_evidence(self, pack: EvidencePack, claim_text: str) -> list[EvidenceMatch]:
        """Match evidence snippets against a claim."""
        if not pack or not pack.snippets:
            return []

        claim_tokens = self._tokenize(claim_text)
        claim_numbers = self._extract_numbers(claim_text)

        matches: list[EvidenceMatch] = []
        for snippet in pack.snippets:
            snippet_text = f"{snippet.title} {snippet.snippet}"
            snippet_tokens = self._tokenize(snippet_text)
            snippet_numbers = self._extract_numbers(snippet_text)

            overlap = claim_tokens.intersection(snippet_tokens)
            number_matches = len(claim_numbers.intersection(snippet_numbers))

            if len(overlap) < self.config.min_overlap_words and number_matches == 0:
                continue

            score = (len(overlap) + number_matches * 2) / max(len(claim_tokens), 1)
            matches.append(
                EvidenceMatch(
                    snippet=snippet,
                    score=score,
                    overlap_words=len(overlap),
                    number_matches=number_matches,
                )
            )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:3]


__all__ = [
    "ClaimCheck",
    "ClaimCheckConfig",
    "EvidenceMatch",
]
