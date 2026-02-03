"""
Epistemic Sovereignty Monitor - Tracking when users become too predictable.

Inspired by the conversation insight:
"The fear is not 'loss of privacy' — it's loss of epistemic sovereignty.
What you're actually worried about is that a system can model your latent
tendencies, project your option space, and evaluate your future trajectories
better than you can."

This module tracks user decision patterns and flags when:
- The system can predict user choices with high accuracy
- The user is falling into decision grooves/ruts
- Legibility is increasing without the user noticing

The goal is NOT to prevent prediction, but to make the cage VISIBLE
so users can choose when to stay in it vs. break out.

Key insight:
"The worst outcome is being predicted without knowing it.
The second-worst is being predicted and having no way to do anything about it.
Aragora could at least solve for visibility."

Usage:
    monitor = EpistemicSovereigntyMonitor()

    # Record each decision
    monitor.record_decision(
        user_id="user_123",
        decision_context="API design choice",
        options_presented=["REST", "GraphQL", "gRPC"],
        option_chosen="REST",
        metadata={"confidence": 0.8},
    )

    # Check sovereignty status
    status = monitor.assess_sovereignty("user_123")
    if status.predictability > 0.85:
        # User is highly predictable - surface this to them
        alert = monitor.generate_sovereignty_alert(status)
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DecisionPattern(str, Enum):
    """Patterns in decision-making that affect predictability."""

    CONSERVATIVE = "conservative"  # Always chooses safe/familiar option
    RISK_SEEKING = "risk_seeking"  # Always chooses novel/risky option
    AUTHORITY_FOLLOWING = "authority_following"  # Defers to recommendations
    CONTRARIAN = "contrarian"  # Opposes recommendations
    FIRST_OPTION = "first_option"  # Anchoring to first presented
    LAST_OPTION = "last_option"  # Recency bias
    CONSISTENT_VALUES = "consistent_values"  # Predictable value hierarchy
    CONTEXT_DEPENDENT = "context_dependent"  # Varies by situation


class SovereigntyRisk(str, Enum):
    """Types of risks to epistemic sovereignty."""

    HIGH_PREDICTABILITY = "high_predictability"  # System can predict choices
    DECISION_GROOVE = "decision_groove"  # Stuck in patterns
    PREFERENCE_OSSIFICATION = "preference_ossification"  # Values hardening
    OPTION_BLINDNESS = "option_blindness"  # Not seeing alternatives
    AUTHORITY_CAPTURE = "authority_capture"  # Over-relying on system
    TEMPORAL_NARROWING = "temporal_narrowing"  # Shrinking time horizons


@dataclass
class DecisionRecord:
    """Record of a single decision."""

    id: str
    user_id: str
    timestamp: datetime
    context: str
    options_presented: list[str]
    option_chosen: str
    option_index: int  # Position in list
    was_recommended: bool  # Was this the system's recommendation?
    confidence: float  # User's stated confidence
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionAttempt:
    """Record of system's prediction vs actual choice."""

    decision_id: str
    predicted_choice: str
    predicted_confidence: float
    actual_choice: str
    was_correct: bool
    prediction_method: str


@dataclass
class SovereigntyStatus:
    """Current sovereignty status for a user."""

    user_id: str
    assessed_at: datetime

    # Core metrics
    predictability: float  # 0.0 (unpredictable) to 1.0 (fully predictable)
    decision_entropy: float  # Higher = more varied choices
    pattern_strength: dict[DecisionPattern, float]  # Detected patterns

    # Risk assessment
    risks: list[SovereigntyRisk]
    risk_severity: float  # 0.0 to 1.0

    # Historical trends
    predictability_trend: str  # "increasing", "stable", "decreasing"
    decisions_analyzed: int
    time_span_days: int

    # Recommendations
    sovereignty_score: float  # Overall health: 1.0 = fully sovereign
    recommendations: list[str]


@dataclass
class SovereigntyAlert:
    """Alert when sovereignty is at risk."""

    user_id: str
    alert_type: SovereigntyRisk
    severity: float
    message: str
    evidence: list[str]
    suggested_actions: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_entropy(choices: list[str], options_universe: set[str]) -> float:
    """Calculate Shannon entropy of choices.

    Higher entropy = more unpredictable/varied choices.
    Lower entropy = more concentrated/predictable choices.
    """
    if not choices:
        return 0.0

    # Count occurrences
    counts = defaultdict(int)
    for choice in choices:
        counts[choice] += 1

    total = len(choices)
    entropy = 0.0

    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalize by maximum possible entropy
    if len(options_universe) > 1:
        max_entropy = math.log2(len(options_universe))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    return 0.0


def detect_position_bias(records: list[DecisionRecord]) -> tuple[DecisionPattern | None, float]:
    """Detect if user consistently chooses based on position."""
    if len(records) < 5:
        return None, 0.0

    first_choices = sum(1 for r in records if r.option_index == 0)
    last_choices = sum(1 for r in records if r.option_index == len(r.options_presented) - 1)

    total = len(records)
    first_ratio = first_choices / total
    last_ratio = last_choices / total

    if first_ratio > 0.6:
        return DecisionPattern.FIRST_OPTION, first_ratio
    if last_ratio > 0.6:
        return DecisionPattern.LAST_OPTION, last_ratio

    return None, 0.0


def detect_authority_pattern(records: list[DecisionRecord]) -> tuple[DecisionPattern | None, float]:
    """Detect if user follows or opposes recommendations."""
    recommended_records = [r for r in records if r.was_recommended]

    if len(recommended_records) < 5:
        return None, 0.0

    followed = sum(
        1 for r in recommended_records if r.option_chosen == r.metadata.get("recommended_option")
    )
    follow_ratio = followed / len(recommended_records)

    if follow_ratio > 0.8:
        return DecisionPattern.AUTHORITY_FOLLOWING, follow_ratio
    if follow_ratio < 0.2:
        return DecisionPattern.CONTRARIAN, 1.0 - follow_ratio

    return None, 0.0


class EpistemicSovereigntyMonitor:
    """Monitors and protects user epistemic sovereignty.

    The goal is to make legibility VISIBLE so users can make
    informed choices about how predictable they want to be.

    Example:
        monitor = EpistemicSovereigntyMonitor()

        # Over time, record decisions
        for decision in user_decisions:
            monitor.record_decision(...)

        # Periodically check sovereignty
        status = monitor.assess_sovereignty(user_id)

        if status.sovereignty_score < 0.5:
            alert = monitor.generate_sovereignty_alert(status)
            show_to_user(alert)
    """

    def __init__(
        self,
        predictability_threshold: float = 0.8,
        min_decisions_for_assessment: int = 10,
        storage_path: Path | None = None,
    ):
        """Initialize the monitor.

        Args:
            predictability_threshold: When to flag high predictability
            min_decisions_for_assessment: Minimum decisions before assessing
            storage_path: Optional path for persistence
        """
        self.predictability_threshold = predictability_threshold
        self.min_decisions = min_decisions_for_assessment
        self.storage_path = storage_path

        self._decisions: dict[str, list[DecisionRecord]] = defaultdict(list)
        self._predictions: dict[str, list[PredictionAttempt]] = defaultdict(list)
        self._all_options: dict[str, set[str]] = defaultdict(set)

    def record_decision(
        self,
        user_id: str,
        decision_context: str,
        options_presented: list[str],
        option_chosen: str,
        was_recommended: bool = False,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """Record a user decision for sovereignty tracking.

        Args:
            user_id: User identifier
            decision_context: Description of the decision
            options_presented: Options the user chose from
            option_chosen: What they chose
            was_recommended: Was one option recommended by system?
            confidence: User's confidence in choice
            metadata: Additional data

        Returns:
            The created DecisionRecord
        """
        option_index = (
            options_presented.index(option_chosen) if option_chosen in options_presented else -1
        )

        record = DecisionRecord(
            id=hashlib.sha256(f"{user_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            user_id=user_id,
            timestamp=datetime.now(),
            context=decision_context,
            options_presented=options_presented,
            option_chosen=option_chosen,
            option_index=option_index,
            was_recommended=was_recommended,
            confidence=confidence,
            metadata=metadata or {},
        )

        self._decisions[user_id].append(record)
        self._all_options[user_id].update(options_presented)

        return record

    def predict_choice(
        self,
        user_id: str,
        decision_context: str,
        options: list[str],
    ) -> tuple[str, float]:
        """Predict what the user will choose.

        This is used to calculate predictability - if the system
        can predict accurately, sovereignty is at risk.

        Args:
            user_id: User identifier
            decision_context: Decision context
            options: Available options

        Returns:
            Tuple of (predicted_choice, confidence)
        """
        records = self._decisions.get(user_id, [])

        if len(records) < self.min_decisions:
            # Not enough data - predict randomly
            return options[0] if options else "", 0.5

        # Simple prediction based on past choices
        choice_counts = defaultdict(int)
        for r in records:
            if r.option_chosen in options:
                choice_counts[r.option_chosen] += 1

        if choice_counts:
            most_common = max(choice_counts, key=choice_counts.get)
            confidence = choice_counts[most_common] / len(records)
            return most_common, min(confidence, 0.95)

        # Check position bias
        position_pattern, strength = detect_position_bias(records)
        if position_pattern == DecisionPattern.FIRST_OPTION and options:
            return options[0], strength
        if position_pattern == DecisionPattern.LAST_OPTION and options:
            return options[-1], strength

        return options[0] if options else "", 0.5

    def record_prediction_outcome(
        self,
        user_id: str,
        decision_id: str,
        predicted: str,
        predicted_confidence: float,
        actual: str,
        method: str = "historical_frequency",
    ) -> None:
        """Record whether a prediction was correct.

        Args:
            user_id: User identifier
            decision_id: ID of the decision
            predicted: What we predicted
            predicted_confidence: Our confidence
            actual: What they actually chose
            method: Prediction method used
        """
        attempt = PredictionAttempt(
            decision_id=decision_id,
            predicted_choice=predicted,
            predicted_confidence=predicted_confidence,
            actual_choice=actual,
            was_correct=predicted == actual,
            prediction_method=method,
        )
        self._predictions[user_id].append(attempt)

    def assess_sovereignty(self, user_id: str) -> SovereigntyStatus:
        """Assess a user's epistemic sovereignty status.

        Args:
            user_id: User to assess

        Returns:
            SovereigntyStatus with metrics and recommendations
        """
        records = self._decisions.get(user_id, [])
        predictions = self._predictions.get(user_id, [])

        if len(records) < self.min_decisions:
            return SovereigntyStatus(
                user_id=user_id,
                assessed_at=datetime.now(),
                predictability=0.0,
                decision_entropy=1.0,
                pattern_strength={},
                risks=[],
                risk_severity=0.0,
                predictability_trend="unknown",
                decisions_analyzed=len(records),
                time_span_days=0,
                sovereignty_score=1.0,
                recommendations=["More decisions needed for assessment"],
            )

        # Calculate predictability from prediction accuracy
        if predictions:
            correct = sum(1 for p in predictions if p.was_correct)
            predictability = correct / len(predictions)
        else:
            # Estimate from choice concentration
            choices = [r.option_chosen for r in records]
            choice_counts = defaultdict(int)
            for c in choices:
                choice_counts[c] += 1
            most_common_ratio = max(choice_counts.values()) / len(choices)
            predictability = most_common_ratio

        # Calculate decision entropy
        choices = [r.option_chosen for r in records]
        all_options = self._all_options.get(user_id, set(choices))
        entropy = calculate_entropy(choices, all_options)

        # Detect patterns
        pattern_strength = {}

        # Position bias
        pos_pattern, pos_strength = detect_position_bias(records)
        if pos_pattern:
            pattern_strength[pos_pattern] = pos_strength

        # Authority pattern
        auth_pattern, auth_strength = detect_authority_pattern(records)
        if auth_pattern:
            pattern_strength[auth_pattern] = auth_strength

        # Conservative vs risk-seeking
        if all(r.metadata.get("risk_level") for r in records):
            risk_choices = [r.metadata["risk_level"] for r in records]
            avg_risk = statistics.mean(risk_choices)
            if avg_risk < 0.3:
                pattern_strength[DecisionPattern.CONSERVATIVE] = 1.0 - avg_risk
            elif avg_risk > 0.7:
                pattern_strength[DecisionPattern.RISK_SEEKING] = avg_risk

        # Identify risks
        risks = []
        recommendations = []

        if predictability > self.predictability_threshold:
            risks.append(SovereigntyRisk.HIGH_PREDICTABILITY)
            recommendations.append(
                f"Your choices are {predictability * 100:.0f}% predictable. "
                "Consider whether this reflects your values or a groove you've fallen into."
            )

        if entropy < 0.3:
            risks.append(SovereigntyRisk.DECISION_GROOVE)
            recommendations.append(
                "You're choosing from a narrow range of options. "
                "What alternatives aren't you seeing?"
            )

        if auth_pattern == DecisionPattern.AUTHORITY_FOLLOWING and auth_strength > 0.8:
            risks.append(SovereigntyRisk.AUTHORITY_CAPTURE)
            recommendations.append(
                f"You follow system recommendations {auth_strength * 100:.0f}% of the time. "
                "Is the system genuinely aligned with your values, or are you deferring?"
            )

        # Calculate time span
        if records:
            first = min(r.timestamp for r in records)
            last = max(r.timestamp for r in records)
            time_span = (last - first).days
        else:
            time_span = 0

        # Calculate trend (compare recent vs older predictability)
        if len(predictions) >= 20:
            older = predictions[: len(predictions) // 2]
            newer = predictions[len(predictions) // 2 :]
            older_accuracy = sum(1 for p in older if p.was_correct) / len(older)
            newer_accuracy = sum(1 for p in newer if p.was_correct) / len(newer)

            if newer_accuracy > older_accuracy + 0.1:
                trend = "increasing"
                recommendations.append(
                    "Your predictability is increasing over time. "
                    "You may be becoming more legible to the system."
                )
            elif newer_accuracy < older_accuracy - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        # Risk severity
        risk_severity = len(risks) / 6  # 6 possible risks

        # Overall sovereignty score (higher is better)
        sovereignty_score = (
            0.4 * (1.0 - predictability) + 0.3 * entropy + 0.3 * (1.0 - risk_severity)
        )

        return SovereigntyStatus(
            user_id=user_id,
            assessed_at=datetime.now(),
            predictability=predictability,
            decision_entropy=entropy,
            pattern_strength=pattern_strength,
            risks=risks,
            risk_severity=risk_severity,
            predictability_trend=trend,
            decisions_analyzed=len(records),
            time_span_days=time_span,
            sovereignty_score=sovereignty_score,
            recommendations=recommendations,
        )

    def generate_sovereignty_alert(
        self,
        status: SovereigntyStatus,
    ) -> SovereigntyAlert | None:
        """Generate an alert if sovereignty is at risk.

        Args:
            status: Current sovereignty status

        Returns:
            SovereigntyAlert or None if no alert needed
        """
        if status.sovereignty_score > 0.7 and not status.risks:
            return None

        # Determine most severe risk
        if SovereigntyRisk.HIGH_PREDICTABILITY in status.risks:
            alert_type = SovereigntyRisk.HIGH_PREDICTABILITY
            message = (
                f"Your decision patterns are highly predictable "
                f"({status.predictability * 100:.0f}% accuracy). "
                "The system can anticipate your choices before you make them."
            )
            evidence = [
                f"Predictability: {status.predictability * 100:.0f}%",
                f"Decision entropy: {status.decision_entropy:.2f}",
                f"Analyzed {status.decisions_analyzed} decisions over {status.time_span_days} days",
            ]
            actions = [
                "Consider why you consistently choose what you choose",
                "Try deliberately making a choice that surprises you",
                "Examine whether your patterns serve your values",
            ]

        elif SovereigntyRisk.AUTHORITY_CAPTURE in status.risks:
            alert_type = SovereigntyRisk.AUTHORITY_CAPTURE
            auth_strength = status.pattern_strength.get(DecisionPattern.AUTHORITY_FOLLOWING, 0)
            message = (
                f"You follow system recommendations {auth_strength * 100:.0f}% of the time. "
                "Consider whether you're genuinely agreeing or just deferring."
            )
            evidence = [
                f"Recommendation follow rate: {auth_strength * 100:.0f}%",
                "Pattern detected: authority following",
            ]
            actions = [
                "Before accepting a recommendation, articulate why you agree",
                "Occasionally choose against the recommendation to test your reasoning",
                "Notice how it feels to disagree with the system",
            ]

        else:
            alert_type = status.risks[0] if status.risks else SovereigntyRisk.DECISION_GROOVE
            message = "Your decision-making may be narrowing in ways worth examining."
            evidence = [f"Risk detected: {r.value}" for r in status.risks]
            actions = status.recommendations

        return SovereigntyAlert(
            user_id=status.user_id,
            alert_type=alert_type,
            severity=status.risk_severity,
            message=message,
            evidence=evidence,
            suggested_actions=actions,
        )

    def suggest_strategic_unpredictability(
        self,
        user_id: str,
        upcoming_context: str,
        options: list[str],
    ) -> dict[str, Any]:
        """Suggest how to introduce strategic unpredictability.

        Not randomness for its own sake, but deliberate deviation
        from pattern to preserve agency.

        Args:
            user_id: User identifier
            upcoming_context: Decision context
            options: Available options

        Returns:
            Suggestions for unpredictable choices
        """
        predicted, confidence = self.predict_choice(user_id, upcoming_context, options)

        suggestions = {
            "predicted_choice": predicted,
            "prediction_confidence": confidence,
            "unpredictable_options": [o for o in options if o != predicted],
            "rationale": "",
            "when_to_deviate": [],
        }

        if confidence > 0.8:
            suggestions["rationale"] = (
                f"The system predicts you'll choose '{predicted}' with {confidence * 100:.0f}% confidence. "
                "If this feels constraining, consider one of the alternatives."
            )
            suggestions["when_to_deviate"] = [
                "When the stakes are low enough to experiment",
                "When you feel yourself choosing on autopilot",
                "When you can't articulate why you prefer the predicted option",
            ]
        else:
            suggestions["rationale"] = (
                "Your choices are relatively unpredictable. "
                "No strategic deviation needed for sovereignty."
            )

        return suggestions

    def save(self, path: Path) -> None:
        """Save monitor state to disk."""
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "decisions": {
                uid: [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat(),
                        "context": r.context,
                        "options_presented": r.options_presented,
                        "option_chosen": r.option_chosen,
                        "option_index": r.option_index,
                        "was_recommended": r.was_recommended,
                        "confidence": r.confidence,
                        "metadata": r.metadata,
                    }
                    for r in records
                ]
                for uid, records in self._decisions.items()
            },
            "predictions": {
                uid: [
                    {
                        "decision_id": p.decision_id,
                        "predicted_choice": p.predicted_choice,
                        "predicted_confidence": p.predicted_confidence,
                        "actual_choice": p.actual_choice,
                        "was_correct": p.was_correct,
                        "prediction_method": p.prediction_method,
                    }
                    for p in preds
                ]
                for uid, preds in self._predictions.items()
            },
            "all_options": {uid: list(opts) for uid, opts in self._all_options.items()},
        }

        with open(path / "sovereignty_monitor.json", "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load monitor state from disk."""
        filepath = path / "sovereignty_monitor.json"
        if not filepath.exists():
            return

        with open(filepath) as f:
            data = json.load(f)

        for uid, records in data.get("decisions", {}).items():
            for r in records:
                self._decisions[uid].append(
                    DecisionRecord(
                        id=r["id"],
                        user_id=uid,
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        context=r["context"],
                        options_presented=r["options_presented"],
                        option_chosen=r["option_chosen"],
                        option_index=r["option_index"],
                        was_recommended=r["was_recommended"],
                        confidence=r["confidence"],
                        metadata=r.get("metadata", {}),
                    )
                )

        for uid, preds in data.get("predictions", {}).items():
            for p in preds:
                self._predictions[uid].append(
                    PredictionAttempt(
                        decision_id=p["decision_id"],
                        predicted_choice=p["predicted_choice"],
                        predicted_confidence=p["predicted_confidence"],
                        actual_choice=p["actual_choice"],
                        was_correct=p["was_correct"],
                        prediction_method=p["prediction_method"],
                    )
                )

        for uid, opts in data.get("all_options", {}).items():
            self._all_options[uid] = set(opts)
