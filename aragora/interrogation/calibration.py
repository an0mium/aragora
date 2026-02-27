"""Interrogation Calibration System.

Learns question effectiveness per domain/category over time.
Feeds back into question prioritization to skip low-value
questions and focus on high-impact clarifications.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionOutcome:
    """Outcome record for a single question asked during interrogation."""

    question_id: str
    category: str
    domain: str
    was_answered: bool
    answer_changed_default: bool = False
    answer_confidence: float = 0.5
    question_priority: float = 0.5


@dataclass
class SessionRecord:
    """Record of one interrogation session for calibration."""

    session_id: str
    domain: str
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    spec_quality: float = 0.0  # 0-1, post-hoc quality assessment


@dataclass
class CategoryEffectiveness:
    """Effectiveness metrics for a question category in a domain."""

    category: str
    domain: str
    times_asked: int = 0
    times_answered: int = 0
    times_changed_default: int = 0
    total_spec_quality: float = 0.0
    sessions_counted: int = 0

    @property
    def answer_rate(self) -> float:
        """Fraction of times the question was answered."""
        return self.times_answered / max(self.times_asked, 1)

    @property
    def change_rate(self) -> float:
        """Fraction of answers that changed the default assumption."""
        return self.times_changed_default / max(self.times_answered, 1)

    @property
    def avg_spec_quality(self) -> float:
        """Average spec quality from sessions where this category appeared."""
        return self.total_spec_quality / max(self.sessions_counted, 1)

    @property
    def value_score(self) -> float:
        """Combined value score. Higher = more valuable category.

        Weight: 30% answer rate (questions people bother answering are useful),
        70% change rate (questions that change defaults reveal important info).
        """
        return 0.3 * self.answer_rate + 0.7 * self.change_rate


@dataclass
class CalibrationSnapshot:
    """Snapshot of calibration state for external consumption."""

    domain: str
    categories: dict[str, CategoryEffectiveness]
    focus_categories: list[str]  # categories with above-average value
    total_sessions: int


class InterrogationCalibrator:
    """Learns question effectiveness and adjusts prioritization.

    Tracks which question categories yield useful answers across
    sessions and provides priority adjustments for the interrogation
    engine.
    """

    def __init__(self) -> None:
        # domain → category → effectiveness
        self._effectiveness: dict[str, dict[str, CategoryEffectiveness]] = defaultdict(dict)
        self._session_count: dict[str, int] = defaultdict(int)

    def record_session(self, record: SessionRecord) -> None:
        """Record outcomes from an interrogation session."""
        self._session_count[record.domain] += 1
        domain_eff = self._effectiveness[record.domain]

        categories_seen: set[str] = set()
        for outcome in record.outcomes:
            cat = outcome.category
            categories_seen.add(cat)

            if cat not in domain_eff:
                domain_eff[cat] = CategoryEffectiveness(category=cat, domain=record.domain)

            eff = domain_eff[cat]
            eff.times_asked += 1
            if outcome.was_answered:
                eff.times_answered += 1
            if outcome.answer_changed_default:
                eff.times_changed_default += 1

        # Add spec quality to all categories seen in this session
        for cat in categories_seen:
            domain_eff[cat].total_spec_quality += record.spec_quality
            domain_eff[cat].sessions_counted += 1

    def get_effectiveness(self, domain: str, category: str) -> CategoryEffectiveness | None:
        """Get effectiveness metrics for a specific domain/category."""
        return self._effectiveness.get(domain, {}).get(category)

    def get_calibration(self, domain: str) -> CalibrationSnapshot:
        """Get full calibration snapshot for a domain."""
        domain_eff = self._effectiveness.get(domain, {})

        # Focus categories: above-average value score
        if domain_eff:
            avg_value = sum(e.value_score for e in domain_eff.values()) / len(domain_eff)
            focus = [cat for cat, eff in domain_eff.items() if eff.value_score > avg_value]
        else:
            focus = []

        return CalibrationSnapshot(
            domain=domain,
            categories=dict(domain_eff),
            focus_categories=focus,
            total_sessions=self._session_count.get(domain, 0),
        )

    def should_ask(self, domain: str, category: str, min_sessions: int = 3) -> bool:
        """Whether a question category is worth asking in this domain.

        Returns True if:
        - Not enough data yet (fewer than min_sessions)
        - Category has above-threshold value score (> 0.15)
        """
        eff = self.get_effectiveness(domain, category)
        if eff is None or eff.sessions_counted < min_sessions:
            return True  # Not enough data, ask by default
        return eff.value_score > 0.15

    def get_priority_adjustment(self, domain: str, category: str) -> float:
        """Get priority multiplier for a question category.

        Returns:
            Multiplier (0.5 - 1.5) to apply to base question priority.
            1.0 = neutral, <1.0 = deprioritize, >1.0 = boost.
        """
        eff = self.get_effectiveness(domain, category)
        if eff is None or eff.sessions_counted < 2:
            return 1.0  # Neutral

        # Scale value_score (typically 0-1) to multiplier (0.5-1.5)
        return 0.5 + eff.value_score

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration state."""
        result: dict[str, Any] = {}
        for domain, cats in self._effectiveness.items():
            result[domain] = {
                "session_count": self._session_count[domain],
                "categories": {
                    cat: {
                        "times_asked": eff.times_asked,
                        "times_answered": eff.times_answered,
                        "times_changed_default": eff.times_changed_default,
                        "value_score": round(eff.value_score, 3),
                        "answer_rate": round(eff.answer_rate, 3),
                        "change_rate": round(eff.change_rate, 3),
                    }
                    for cat, eff in cats.items()
                },
            }
        return result
