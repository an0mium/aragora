"""
Domain classification for debate questions.

Provides keyword-based domain detection for selecting appropriate personas
and context. Extracted from PromptBuilder for improved modularity and testability.
"""

from __future__ import annotations

import re
from typing import Set


# Domain keyword mappings
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "philosophical": [
        # Core philosophical concepts
        "meaning",
        "meaningful",
        "life",
        "purpose",
        "existence",
        "happiness",
        "soul",
        "consciousness",
        "free will",
        "morality",
        "good life",
        "philosophy",
        "wisdom",
        "death",
        "love",
        "truth",
        "human condition",
        "fulfillment",
        "wellbeing",
        "flourishing",
        # Theological/Religious keywords
        "heaven",
        "hell",
        "afterlife",
        "divine",
        "god",
        "theological",
        "theology",
        "religious",
        "faith",
        "spiritual",
        "prayer",
        "scripture",
        "bible",
        "sin",
        "redemption",
        "salvation",
        "eternal",
        "sacred",
        "holy",
        "angel",
        "fetus",
        "abortion",
    ],
    "ethics": [
        "should",
        "ethical",
        "moral",
        "right",
        "wrong",
        "justice",
        "fair",
        "harm",
        "good or bad",
        "bad or good",
    ],
    "technical": [
        "code",
        "api",
        "software",
        "architecture",
        "database",
        "security",
        "testing",
        "function",
        "class",
        "microservice",
        "deployment",
        "infrastructure",
        "programming",
        "algorithm",
        "backend",
        "frontend",
    ],
}


def word_match(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears as a whole word in text.

    Uses word boundary regex to avoid substring matches
    (e.g., "api" shouldn't match "capitalism").

    Args:
        text: The text to search in (should be lowercase).
        keywords: List of keywords to search for.

    Returns:
        True if any keyword matches as a whole word.
    """
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True
    return False


def detect_question_domain_keywords(question: str) -> str:
    """Keyword-based domain detection (fallback when LLM unavailable).

    Analyzes the question text to determine its domain category
    based on keyword matching.

    Args:
        question: The debate question or task description.

    Returns:
        Domain category: 'philosophical', 'ethics', 'technical', or 'general'.
    """
    lower = question.lower()

    # Philosophical/Life/Theological domains - use philosophical personas
    if word_match(lower, DOMAIN_KEYWORDS["philosophical"]):
        return "philosophical"

    # Ethics domain - questions about right/wrong, good/bad, should/shouldn't
    if word_match(lower, DOMAIN_KEYWORDS["ethics"]):
        return "ethics"

    # Technical domain - software, code, architecture
    if word_match(lower, DOMAIN_KEYWORDS["technical"]):
        return "technical"

    return "general"


def get_domain_keywords(domain: str) -> Set[str]:
    """Get the keywords associated with a domain.

    Args:
        domain: The domain name ('philosophical', 'ethics', 'technical').

    Returns:
        Set of keywords for the domain, or empty set if unknown.
    """
    return set(DOMAIN_KEYWORDS.get(domain, []))


def classify_by_keywords(text: str) -> dict[str, bool]:
    """Classify text against all domain keyword sets.

    Useful for understanding which domains a question might span.

    Args:
        text: The text to classify.

    Returns:
        Dict mapping domain names to whether they matched.
    """
    lower = text.lower()
    return {domain: word_match(lower, keywords) for domain, keywords in DOMAIN_KEYWORDS.items()}
