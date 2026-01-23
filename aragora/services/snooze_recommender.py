"""
Snooze Recommender Service.

Intelligent snooze timing suggestions based on:
- Sender history and typical response patterns
- User's calendar availability
- Email priority and urgency
- Time of day and day of week patterns
- Priority decay estimation

Usage:
    from aragora.services.snooze_recommender import SnoozeRecommender

    recommender = SnoozeRecommender(
        sender_history=sender_service,
        calendar_service=calendar_service,
    )

    # Get snooze recommendations
    suggestions = await recommender.recommend_snooze(email, priority_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.connectors.enterprise.communication.models import EmailMessage
    from aragora.services.email_prioritization import EmailPriorityResult
    from aragora.services.sender_history import SenderHistoryService

logger = logging.getLogger(__name__)


class SnoozeReason(str, Enum):
    """Reason for snooze suggestion."""

    CALENDAR_FREE = "calendar_free"  # User has free time then
    SENDER_PATTERN = "sender_pattern"  # Based on sender history
    WORK_HOURS = "work_hours"  # Optimal work hours
    PRIORITY_DECAY = "priority_decay"  # Can wait based on priority
    WEEKEND_SKIP = "weekend_skip"  # Skip to weekday
    END_OF_DAY = "end_of_day"  # Review at EOD
    TOMORROW_MORNING = "tomorrow_morning"  # Fresh start


@dataclass
class SnoozeSuggestion:
    """A snooze time suggestion."""

    snooze_until: datetime
    reason: SnoozeReason
    label: str  # Human-readable label
    confidence: float = 0.8
    rationale: str = ""
    is_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snooze_until": self.snooze_until.isoformat(),
            "reason": self.reason.value,
            "label": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "is_recommended": self.is_recommended,
        }


@dataclass
class SnoozeRecommendation:
    """Complete snooze recommendation with multiple options."""

    email_id: str
    suggestions: List[SnoozeSuggestion]
    recommended: Optional[SnoozeSuggestion] = None
    priority_level: int = 3
    can_safely_snooze: bool = True
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "priority_level": self.priority_level,
            "can_safely_snooze": self.can_safely_snooze,
            "warning": self.warning,
        }


@dataclass
class WorkSchedule:
    """User's work schedule configuration."""

    work_start: time = field(default_factory=lambda: time(9, 0))
    work_end: time = field(default_factory=lambda: time(17, 0))
    work_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    timezone: str = "UTC"
    prefer_morning: bool = True  # Prefer morning for important emails


class SnoozeRecommender:
    """
    Intelligent snooze recommendation service.

    Suggests optimal snooze times based on context,
    calendar, and sender patterns.
    """

    def __init__(
        self,
        sender_history: Optional[SenderHistoryService] = None,
        calendar_service: Optional[Any] = None,
        work_schedule: Optional[WorkSchedule] = None,
    ):
        self.sender_history = sender_history
        self.calendar_service = calendar_service
        self.work_schedule = work_schedule or WorkSchedule()

    async def recommend_snooze(
        self,
        email: EmailMessage,
        priority_result: Optional[EmailPriorityResult] = None,
        max_suggestions: int = 4,
    ) -> SnoozeRecommendation:
        """
        Generate snooze recommendations for an email.

        Args:
            email: Email to snooze
            priority_result: Optional priority scoring result
            max_suggestions: Maximum number of suggestions

        Returns:
            SnoozeRecommendation with options
        """
        email_id = getattr(email, "id", "unknown")
        priority = priority_result.priority.value if priority_result else 3

        suggestions: List[SnoozeSuggestion] = []
        now = datetime.now()

        # Check if email can be safely snoozed
        can_snooze = True
        warning = None

        if priority <= 2:  # Critical or High
            can_snooze = True  # Still allow snooze but warn
            warning = "This is a high-priority email. Consider responding soon."

        # Generate suggestions based on different factors

        # 1. Quick options (Later today, Tomorrow morning)
        suggestions.extend(self._get_quick_suggestions(now, priority))

        # 2. Calendar-based suggestions
        if self.calendar_service:
            calendar_suggestions = await self._get_calendar_suggestions(now)
            suggestions.extend(calendar_suggestions)

        # 3. Sender pattern-based suggestions
        if self.sender_history:
            sender = getattr(email, "from_address", getattr(email, "sender", ""))
            pattern_suggestions = await self._get_sender_pattern_suggestions(sender, now)
            suggestions.extend(pattern_suggestions)

        # 4. Priority-decay based suggestions
        decay_suggestions = self._get_priority_decay_suggestions(priority, now)
        suggestions.extend(decay_suggestions)

        # 5. Work schedule based suggestions
        work_suggestions = self._get_work_schedule_suggestions(now)
        suggestions.extend(work_suggestions)

        # Deduplicate and sort by time
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions.sort(key=lambda x: x.snooze_until)

        # Limit suggestions
        suggestions = suggestions[:max_suggestions]

        # Pick recommended
        recommended = self._pick_recommended(suggestions, priority)
        if recommended:
            recommended.is_recommended = True

        return SnoozeRecommendation(
            email_id=email_id,
            suggestions=suggestions,
            recommended=recommended,
            priority_level=priority,
            can_safely_snooze=can_snooze,
            warning=warning,
        )

    def _get_quick_suggestions(
        self,
        now: datetime,
        priority: int,
    ) -> List[SnoozeSuggestion]:
        """Get quick snooze options."""
        suggestions = []

        # Later today (2-4 hours)
        if now.hour < 16:  # Before 4 PM
            later_today = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
            suggestions.append(
                SnoozeSuggestion(
                    snooze_until=later_today,
                    reason=SnoozeReason.END_OF_DAY,
                    label="Later today",
                    confidence=0.7,
                    rationale="Quick review later in the day",
                )
            )

        # Tomorrow morning
        tomorrow_morning = self._next_work_morning(now)
        suggestions.append(
            SnoozeSuggestion(
                snooze_until=tomorrow_morning,
                reason=SnoozeReason.TOMORROW_MORNING,
                label="Tomorrow morning",
                confidence=0.85,
                rationale="Fresh start tomorrow",
            )
        )

        # This weekend (if weekday)
        if now.weekday() < 5 and priority >= 4:  # Low priority
            next_saturday = now + timedelta(days=(5 - now.weekday()) % 7 or 7)
            next_saturday = next_saturday.replace(hour=10, minute=0, second=0, microsecond=0)
            suggestions.append(
                SnoozeSuggestion(
                    snooze_until=next_saturday,
                    reason=SnoozeReason.WEEKEND_SKIP,
                    label="This weekend",
                    confidence=0.6,
                    rationale="Review during less busy time",
                )
            )

        # Next week
        next_monday = now + timedelta(days=(7 - now.weekday()) % 7 or 7)
        next_monday = next_monday.replace(
            hour=self.work_schedule.work_start.hour, minute=0, second=0, microsecond=0
        )
        if priority >= 3:  # Medium or lower
            suggestions.append(
                SnoozeSuggestion(
                    snooze_until=next_monday,
                    reason=SnoozeReason.PRIORITY_DECAY,
                    label="Next week",
                    confidence=0.5,
                    rationale="Can wait until next week",
                )
            )

        return suggestions

    async def _get_calendar_suggestions(
        self,
        now: datetime,
    ) -> List[SnoozeSuggestion]:
        """Get calendar-based snooze suggestions."""
        suggestions = []

        try:
            # Get calendar availability (mock implementation)
            # In production, this would query Google Calendar
            free_slots = await self._find_free_slots(now, days_ahead=3)

            for slot in free_slots[:2]:
                suggestions.append(
                    SnoozeSuggestion(
                        snooze_until=slot,
                        reason=SnoozeReason.CALENDAR_FREE,
                        label=self._format_slot_label(slot),
                        confidence=0.9,
                        rationale=f"You have free time at {slot.strftime('%H:%M')}",
                    )
                )

        except Exception as e:
            logger.debug(f"Calendar lookup failed: {e}")

        return suggestions

    async def _find_free_slots(
        self,
        now: datetime,
        days_ahead: int = 3,
    ) -> List[datetime]:
        """Find free slots in calendar."""
        # Mock implementation - would integrate with calendar API
        slots = []

        for day_offset in range(days_ahead):
            date = now + timedelta(days=day_offset)

            # Skip weekends
            if date.weekday() not in self.work_schedule.work_days:
                continue

            # Morning slot
            morning = date.replace(
                hour=self.work_schedule.work_start.hour, minute=0, second=0, microsecond=0
            )
            if morning > now:
                slots.append(morning)

            # After lunch slot
            afternoon = date.replace(hour=14, minute=0, second=0, microsecond=0)
            if afternoon > now:
                slots.append(afternoon)

        return slots

    async def _get_sender_pattern_suggestions(
        self,
        sender_email: str,
        now: datetime,
    ) -> List[SnoozeSuggestion]:
        """Get suggestions based on sender patterns."""
        suggestions: List[SnoozeSuggestion] = []

        if not self.sender_history or not sender_email:
            return suggestions

        try:
            # Get sender stats
            stats = await self.sender_history.get_stats(sender_email)  # type: ignore[call-arg]
            if not stats:
                return suggestions

            # If user typically responds within X hours, suggest that
            avg_response: float = getattr(stats, "avg_response_time_hours", 24.0)  # type: ignore[assignment]

            if avg_response < 4:
                # Quick responder - suggest soon
                snooze_time = now + timedelta(hours=1)
                suggestions.append(
                    SnoozeSuggestion(
                        snooze_until=snooze_time,
                        reason=SnoozeReason.SENDER_PATTERN,
                        label="In 1 hour",
                        confidence=0.75,
                        rationale=f"You typically respond to {sender_email.split('@')[0]} quickly",
                    )
                )
            elif avg_response > 48:
                # Slow responder - can wait
                snooze_time = now + timedelta(days=2)
                suggestions.append(
                    SnoozeSuggestion(
                        snooze_until=snooze_time,
                        reason=SnoozeReason.SENDER_PATTERN,
                        label="In 2 days",
                        confidence=0.7,
                        rationale="Emails from this sender can typically wait",
                    )
                )

        except Exception as e:
            logger.debug(f"Sender pattern lookup failed: {e}")

        return suggestions

    def _get_priority_decay_suggestions(
        self,
        priority: int,
        now: datetime,
    ) -> List[SnoozeSuggestion]:
        """Get suggestions based on priority decay."""
        suggestions = []

        # Priority decay windows
        decay_windows = {
            1: timedelta(hours=2),  # Critical - short snooze only
            2: timedelta(hours=4),  # High - half day max
            3: timedelta(days=1),  # Medium - up to a day
            4: timedelta(days=3),  # Low - several days
            5: timedelta(days=7),  # Defer - up to a week
        }

        window = decay_windows.get(priority, timedelta(days=1))
        snooze_time = now + window

        # Adjust to work hours
        snooze_time = self._adjust_to_work_hours(snooze_time)

        label_map = {
            1: "Soon (2 hours)",
            2: "Later today",
            3: "Tomorrow",
            4: "In a few days",
            5: "Next week",
        }

        suggestions.append(
            SnoozeSuggestion(
                snooze_until=snooze_time,
                reason=SnoozeReason.PRIORITY_DECAY,
                label=label_map.get(priority, "Later"),
                confidence=0.65,
                rationale=f"Based on priority level {priority}",
            )
        )

        return suggestions

    def _get_work_schedule_suggestions(
        self,
        now: datetime,
    ) -> List[SnoozeSuggestion]:
        """Get suggestions based on work schedule."""
        suggestions = []

        # End of work day
        end_of_day = now.replace(
            hour=self.work_schedule.work_end.hour - 1, minute=0, second=0, microsecond=0
        )
        if end_of_day > now:
            suggestions.append(
                SnoozeSuggestion(
                    snooze_until=end_of_day,
                    reason=SnoozeReason.END_OF_DAY,
                    label="End of day",
                    confidence=0.7,
                    rationale="Review before leaving for the day",
                )
            )

        # Start of next work day
        next_start = self._next_work_morning(now)
        if next_start not in [s.snooze_until for s in suggestions]:
            suggestions.append(
                SnoozeSuggestion(
                    snooze_until=next_start,
                    reason=SnoozeReason.WORK_HOURS,
                    label="Start of day",
                    confidence=0.8,
                    rationale="Review at the start of work hours",
                )
            )

        return suggestions

    def _next_work_morning(self, now: datetime) -> datetime:
        """Get next work morning."""
        next_day = now + timedelta(days=1)
        next_day = next_day.replace(
            hour=self.work_schedule.work_start.hour, minute=0, second=0, microsecond=0
        )

        # Skip to Monday if weekend
        while next_day.weekday() not in self.work_schedule.work_days:
            next_day += timedelta(days=1)

        return next_day

    def _adjust_to_work_hours(self, dt: datetime) -> datetime:
        """Adjust datetime to work hours."""
        # If weekend, move to Monday
        while dt.weekday() not in self.work_schedule.work_days:
            dt += timedelta(days=1)

        # If before work hours, set to start
        if dt.time() < self.work_schedule.work_start:
            dt = dt.replace(
                hour=self.work_schedule.work_start.hour, minute=0, second=0, microsecond=0
            )
        # If after work hours, move to next work day
        elif dt.time() > self.work_schedule.work_end:
            dt += timedelta(days=1)
            while dt.weekday() not in self.work_schedule.work_days:
                dt += timedelta(days=1)
            dt = dt.replace(
                hour=self.work_schedule.work_start.hour, minute=0, second=0, microsecond=0
            )

        return dt

    def _format_slot_label(self, dt: datetime) -> str:
        """Format datetime as a label."""
        now = datetime.now()

        if dt.date() == now.date():
            return f"Today at {dt.strftime('%H:%M')}"
        elif dt.date() == (now + timedelta(days=1)).date():
            return f"Tomorrow at {dt.strftime('%H:%M')}"
        else:
            return dt.strftime("%A at %H:%M")

    def _deduplicate_suggestions(
        self,
        suggestions: List[SnoozeSuggestion],
    ) -> List[SnoozeSuggestion]:
        """Remove duplicate suggestions (same time)."""
        seen_times = set()
        unique = []

        for s in suggestions:
            # Round to nearest 30 minutes for dedup
            rounded = s.snooze_until.replace(
                minute=(s.snooze_until.minute // 30) * 30, second=0, microsecond=0
            )
            if rounded not in seen_times:
                seen_times.add(rounded)
                unique.append(s)

        return unique

    def _pick_recommended(
        self,
        suggestions: List[SnoozeSuggestion],
        priority: int,
    ) -> Optional[SnoozeSuggestion]:
        """Pick the recommended suggestion."""
        if not suggestions:
            return None

        # For high priority, pick soonest
        if priority <= 2:
            return min(suggestions, key=lambda x: x.snooze_until)

        # For medium priority, pick calendar-based if available
        calendar_suggestions = [s for s in suggestions if s.reason == SnoozeReason.CALENDAR_FREE]
        if calendar_suggestions:
            return calendar_suggestions[0]

        # Otherwise pick highest confidence
        return max(suggestions, key=lambda x: x.confidence)


# Convenience function
async def get_snooze_suggestions(
    email: EmailMessage,
    priority: int = 3,
) -> List[Dict[str, Any]]:
    """
    Quick snooze suggestions for an email.

    Args:
        email: Email to snooze
        priority: Priority level (1-5)

    Returns:
        List of suggestion dictionaries
    """
    recommender = SnoozeRecommender()

    # Create mock priority result
    from aragora.services.email_prioritization import (
        EmailPriority,
        EmailPriorityResult,
        ScoringTier,
    )

    priority_result = EmailPriorityResult(
        email_id=getattr(email, "id", "unknown"),
        priority=EmailPriority(priority),
        confidence=0.8,
        tier_used=ScoringTier.TIER_1_RULES,
        rationale="",
    )

    recommendation = await recommender.recommend_snooze(email, priority_result)
    return [s.to_dict() for s in recommendation.suggestions]
