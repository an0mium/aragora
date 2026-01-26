"""
Action Item Extraction Service.

Intelligent extraction of action items from email content:
- Extract explicit action items ("please review", "send me", "by Friday")
- Parse deadlines from natural language
- Identify task owners from context
- Support urgency classification
- Track completion status

Features:
- Pattern-based extraction for common action phrases
- Natural language deadline parsing
- Owner/assignee detection
- Priority/urgency scoring
- Integration ready for project management tools

Usage:
    from aragora.services.action_item_extractor import ActionItemExtractor

    extractor = ActionItemExtractor()

    # Extract action items from email
    result = await extractor.extract_action_items(email)
    for item in result.action_items:
        print(f"- {item.description}")
        print(f"  Deadline: {item.deadline}")
        print(f"  Assignee: {item.assignee}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Pattern, Set, Tuple

if TYPE_CHECKING:
    from aragora.connectors.enterprise.communication.models import EmailMessage

logger = logging.getLogger(__name__)


class ActionItemPriority(Enum):
    """Priority levels for action items."""

    CRITICAL = 1  # Immediate action required
    HIGH = 2  # Important, act soon
    MEDIUM = 3  # Standard priority
    LOW = 4  # Can wait


class ActionItemStatus(Enum):
    """Status of an action item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class ActionType(Enum):
    """Types of action items."""

    REVIEW = "review"  # Review document/code/proposal
    RESPOND = "respond"  # Send a response
    SEND = "send"  # Send something
    PROVIDE = "provide"  # Provide information/feedback
    SCHEDULE = "schedule"  # Schedule meeting/call
    APPROVE = "approve"  # Approve something
    COMPLETE = "complete"  # Complete a task
    UPDATE = "update"  # Update status/document
    FOLLOW_UP = "follow_up"  # Follow up on something
    CONFIRM = "confirm"  # Confirm attendance/information
    DECISION = "decision"  # Make a decision
    CREATE = "create"  # Create document/resource
    OTHER = "other"


@dataclass
class ActionItem:
    """Represents an extracted action item."""

    id: str
    description: str
    action_type: ActionType
    priority: ActionItemPriority
    status: ActionItemStatus = ActionItemStatus.PENDING

    # Context
    source_email_id: str = ""
    source_text: str = ""  # Original text that triggered detection
    context: str = ""  # Surrounding context

    # Assignment
    assignee_email: Optional[str] = None
    assignee_name: Optional[str] = None
    requester_email: Optional[str] = None
    requester_name: Optional[str] = None

    # Timing
    deadline: Optional[datetime] = None
    deadline_text: Optional[str] = None  # Original deadline text
    is_urgent: bool = False
    is_explicit_deadline: bool = False

    # Metadata
    confidence: float = 0.0
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    # Integration
    external_task_id: Optional[str] = None  # ID in external system (Jira, Asana, etc.)
    external_system: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "description": self.description,
            "action_type": self.action_type.value,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "status": self.status.value,
            "source_email_id": self.source_email_id,
            "source_text": self.source_text,
            "context": self.context,
            "assignee_email": self.assignee_email,
            "assignee_name": self.assignee_name,
            "requester_email": self.requester_email,
            "requester_name": self.requester_name,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "deadline_text": self.deadline_text,
            "is_urgent": self.is_urgent,
            "is_explicit_deadline": self.is_explicit_deadline,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "external_task_id": self.external_task_id,
            "external_system": self.external_system,
        }


@dataclass
class ExtractionResult:
    """Result of action item extraction from an email."""

    email_id: str
    action_items: List[ActionItem]
    total_count: int
    high_priority_count: int
    has_deadlines: bool
    earliest_deadline: Optional[datetime] = None
    extraction_confidence: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_id": self.email_id,
            "action_items": [item.to_dict() for item in self.action_items],
            "total_count": self.total_count,
            "high_priority_count": self.high_priority_count,
            "has_deadlines": self.has_deadlines,
            "earliest_deadline": (
                self.earliest_deadline.isoformat() if self.earliest_deadline else None
            ),
            "extraction_confidence": self.extraction_confidence,
            "processing_time_ms": self.processing_time_ms,
        }


# Action item patterns with types and weights
ACTION_PATTERNS = {
    ActionType.REVIEW: [
        (r"\bplease\s+review\b", 0.9),
        (r"\bcan\s+you\s+review\b", 0.85),
        (r"\bneed\s+(you\s+to\s+|your\s+)?review\b", 0.85),
        (r"\bkindly\s+review\b", 0.85),
        (r"\breview\s+and\s+(approve|comment|feedback)\b", 0.9),
        (r"\btake\s+a\s+look\s+at\b", 0.7),
        (r"\bcheck\s+(this|the|my)\b", 0.6),
        (r"\beyes\s+on\s+(this|the)\b", 0.7),
    ],
    ActionType.RESPOND: [
        (r"\bplease\s+(respond|reply|get\s+back)\b", 0.9),
        (r"\blet\s+me\s+know\b", 0.7),
        (r"\bawait(ing)?\s+(your|a)\s+(response|reply)\b", 0.85),
        (r"\bneed\s+(your\s+)?(input|feedback|thoughts)\b", 0.8),
        (r"\bwhat\s+do\s+you\s+think\b", 0.6),
        (r"\bget\s+back\s+to\s+me\b", 0.85),
    ],
    ActionType.SEND: [
        (r"\bplease\s+send\b", 0.9),
        (r"\bcan\s+you\s+send\b", 0.85),
        (r"\bsend\s+(me|us|over|along)\b", 0.8),
        (r"\bforward\s+(me|us|this)\b", 0.8),
        (r"\bshare\s+(the|your|with)\b", 0.7),
    ],
    ActionType.PROVIDE: [
        (r"\bplease\s+provide\b", 0.9),
        (r"\bcan\s+you\s+provide\b", 0.85),
        (r"\bprovide\s+(me|us|your)\b", 0.8),
        (r"\bneed\s+(the|this|your)\s+\w+\s+(from\s+you|asap|urgently)", 0.8),
        (r"\bgive\s+(me|us)\s+(an\s+)?update\b", 0.8),
    ],
    ActionType.SCHEDULE: [
        (r"\bschedule\s+(a|the)\s+(call|meeting|time)\b", 0.9),
        (r"\bset\s+up\s+(a\s+)?(call|meeting|time)\b", 0.85),
        (r"\bfind\s+(a\s+)?time\s+to\b", 0.7),
        (r"\bblock\s+(some\s+)?time\b", 0.7),
        (r"\bput\s+(on|in)\s+(the\s+)?calendar\b", 0.8),
    ],
    ActionType.APPROVE: [
        (r"\bplease\s+approve\b", 0.9),
        (r"\bneed\s+(your\s+)?approval\b", 0.9),
        (r"\bawait(ing)?\s+(your\s+)?approval\b", 0.9),
        (r"\bsign\s+off\s+on\b", 0.85),
        (r"\bgive\s+(the\s+)?go[-\s]ahead\b", 0.8),
    ],
    ActionType.COMPLETE: [
        (r"\bplease\s+complete\b", 0.9),
        (r"\bfinish\s+(the|this|up)\b", 0.8),
        (r"\bwrap\s+(this\s+)?up\b", 0.7),
        (r"\bget\s+(this|it)\s+done\b", 0.8),
        (r"\bdeliver\s+(by|before)\b", 0.8),
    ],
    ActionType.UPDATE: [
        (r"\bplease\s+update\b", 0.9),
        (r"\bupdate\s+(me|us|the)\b", 0.8),
        (r"\bkeep\s+(me|us)\s+(posted|updated|informed)\b", 0.7),
        (r"\bstatus\s+update\b", 0.6),
    ],
    ActionType.FOLLOW_UP: [
        (r"\bfollow\s+up\s+(on|with)\b", 0.9),
        (r"\bplease\s+follow\s+up\b", 0.9),
        (r"\bcheck\s+(back\s+)?(in|on)\b", 0.7),
        (r"\bloop\s+back\b", 0.7),
        (r"\bcircle\s+back\b", 0.7),
    ],
    ActionType.CONFIRM: [
        (r"\bplease\s+confirm\b", 0.9),
        (r"\bcan\s+you\s+confirm\b", 0.85),
        (r"\bconfirm\s+(your|the|this)\b", 0.8),
        (r"\bverify\s+(that|this)\b", 0.7),
        (r"\backnowledge\s+(receipt|this)\b", 0.7),
    ],
    ActionType.DECISION: [
        (r"\bmake\s+a\s+decision\b", 0.9),
        (r"\bdecide\s+(on|whether)\b", 0.85),
        (r"\bneed\s+(a\s+)?decision\b", 0.9),
        (r"\bweigh\s+in\s+on\b", 0.7),
        (r"\bpick\s+(one|between)\b", 0.6),
    ],
    ActionType.CREATE: [
        (r"\bplease\s+create\b", 0.9),
        (r"\bcan\s+you\s+create\b", 0.85),
        (r"\bdraft\s+(a|the)\b", 0.8),
        (r"\bprepare\s+(a|the)\b", 0.8),
        (r"\bput\s+together\b", 0.7),
        (r"\bwrite\s+up\b", 0.7),
    ],
}

# Urgency indicators
URGENCY_PATTERNS = [
    (r"\b(asap|a\.s\.a\.p\.)\b", 0.95),
    (r"\burgent(ly)?\b", 0.95),
    (r"\bcritical\b", 0.9),
    (r"\bimmediately\b", 0.95),
    (r"\bas\s+soon\s+as\s+possible\b", 0.9),
    (r"\btop\s+priority\b", 0.9),
    (r"\btime[-\s]?sensitive\b", 0.85),
    (r"\bdeadline\s+(approaching|today|tomorrow)\b", 0.9),
    (r"\beod\b", 0.8),  # End of day
    (r"\bcob\b", 0.8),  # Close of business
    (r"\bblocker\b", 0.85),
    (r"\bblocking\b", 0.8),
]

# Deadline patterns
DEADLINE_PATTERNS = [
    # Explicit deadlines
    (r"by\s+(\w+day)", "weekday"),
    (r"by\s+end\s+of\s+(day|week|month)", "relative_end"),
    (r"by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", "date"),
    (r"due\s+(on\s+)?(\w+day)", "weekday"),
    (r"due\s+(on\s+)?(\d{1,2}[/-]\d{1,2})", "date"),
    (r"deadline:?\s*(\d{1,2}[/-]\d{1,2})", "date"),
    (r"no\s+later\s+than\s+(\w+)", "relative"),
    # Relative deadlines
    (r"(today|tonight)\b", "today"),
    (r"(tomorrow)\b", "tomorrow"),
    (r"(this|next)\s+(week|monday|tuesday|wednesday|thursday|friday)", "relative_week"),
    (r"end\s+of\s+(business|day|week)", "eod_eow"),
    (r"within\s+(\d+)\s+(hour|day|week)s?", "within"),
    (r"in\s+(\d+)\s+(hour|day|week)s?", "in_time"),
    # Specific times
    (r"by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", "time"),
    (r"before\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", "time"),
]


class ActionItemExtractor:
    """
    Intelligent action item extraction from emails.

    Uses pattern matching to identify action items, extract deadlines,
    and determine assignees from email content.
    """

    def __init__(
        self,
        user_email: Optional[str] = None,
        user_name: Optional[str] = None,
    ):
        """
        Initialize action item extractor.

        Args:
            user_email: Current user's email for assignment detection
            user_name: Current user's name
        """
        self.user_email = user_email
        self.user_name = user_name

        # Compile patterns
        self._compiled_action_patterns: Dict[ActionType, List[Tuple[Pattern[str], float]]] = {}
        self._compiled_urgency_patterns: List[Tuple[Pattern[str], float]] = []
        self._compile_patterns()

        self._action_counter = 0

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for action_type, patterns in ACTION_PATTERNS.items():
            self._compiled_action_patterns[action_type] = [
                (re.compile(p[0], re.IGNORECASE), p[1]) for p in patterns
            ]

        self._compiled_urgency_patterns = [
            (re.compile(p[0], re.IGNORECASE), p[1]) for p in URGENCY_PATTERNS
        ]

    def _generate_action_id(self, email_id: str) -> str:
        """Generate unique action item ID."""
        self._action_counter += 1
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"action_{email_id[:8]}_{timestamp}_{self._action_counter}"

    async def extract_action_items(
        self,
        email: EmailMessage,
        extract_deadlines: bool = True,
        detect_assignees: bool = True,
    ) -> ExtractionResult:
        """
        Extract action items from an email.

        Args:
            email: Email message to analyze
            extract_deadlines: Whether to extract deadline information
            detect_assignees: Whether to detect assignees

        Returns:
            ExtractionResult with extracted action items
        """
        import time

        start_time = time.time()

        email_id = getattr(email, "id", str(hash(str(email))))
        subject = getattr(email, "subject", "")
        body = getattr(email, "body_text", getattr(email, "body", getattr(email, "snippet", "")))
        sender = getattr(
            email, "from_address", getattr(email, "sender", getattr(email, "from_", ""))
        )
        to_addresses = getattr(email, "to_addresses", [])

        # Combine text for analysis
        full_text = f"{subject}\n{body}"

        # Split into sentences for context
        sentences = self._split_sentences(full_text)

        action_items: List[ActionItem] = []
        seen_descriptions: Set[str] = set()

        for sentence in sentences:
            # Check each action type pattern
            for action_type, patterns in self._compiled_action_patterns.items():
                for regex, weight in patterns:
                    match = regex.search(sentence)
                    if match:
                        # Extract action item description
                        description = self._extract_description(sentence, match)

                        # Deduplicate similar descriptions
                        desc_key = description.lower()[:50]
                        if desc_key in seen_descriptions:
                            continue
                        seen_descriptions.add(desc_key)

                        # Determine urgency
                        is_urgent, urgency_score = self._check_urgency(sentence)

                        # Calculate priority
                        priority = self._calculate_priority(weight, urgency_score, action_type)

                        # Extract deadline
                        deadline = None
                        deadline_text = None
                        is_explicit = False
                        if extract_deadlines:
                            deadline, deadline_text, is_explicit = self._extract_deadline(
                                sentence, full_text
                            )

                        # Detect assignee
                        assignee_email = None
                        assignee_name = None
                        if detect_assignees:
                            assignee_email, assignee_name = self._detect_assignee(
                                sentence, to_addresses, sender
                            )

                        # Create action item
                        action_item = ActionItem(
                            id=self._generate_action_id(email_id),
                            description=description,
                            action_type=action_type,
                            priority=priority,
                            source_email_id=email_id,
                            source_text=sentence,
                            context=self._get_context(sentences, sentence),
                            assignee_email=assignee_email,
                            assignee_name=assignee_name,
                            requester_email=sender,
                            deadline=deadline,
                            deadline_text=deadline_text,
                            is_urgent=is_urgent,
                            is_explicit_deadline=is_explicit,
                            confidence=weight,
                            tags=self._extract_tags(sentence, action_type),
                        )

                        action_items.append(action_item)
                        break  # One action per sentence

        # Calculate statistics
        high_priority_count = sum(
            1
            for item in action_items
            if item.priority in (ActionItemPriority.CRITICAL, ActionItemPriority.HIGH)
        )
        has_deadlines = any(item.deadline for item in action_items)
        earliest_deadline = None
        if has_deadlines:
            deadlines = [item.deadline for item in action_items if item.deadline]
            earliest_deadline = min(deadlines)

        avg_confidence = (
            sum(item.confidence for item in action_items) / len(action_items)
            if action_items
            else 0.0
        )

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            email_id=email_id,
            action_items=action_items,
            total_count=len(action_items),
            high_priority_count=high_priority_count,
            has_deadlines=has_deadlines,
            earliest_deadline=earliest_deadline,
            extraction_confidence=avg_confidence,
            processing_time_ms=processing_time,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        # Handle common abbreviations
        text = re.sub(r"(\w)\.\s+(\w)", r"\1. \2", text)

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        # Clean and filter
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _extract_description(self, sentence: str, match: re.Match[str]) -> str:
        """Extract clean description from matched sentence."""
        # Get the matched portion and surrounding context
        start = match.start()
        end = min(len(sentence), match.end() + 100)

        description = sentence[start:end].strip()

        # Clean up
        description = re.sub(r"\s+", " ", description)
        description = description.rstrip(".,;:")

        # Capitalize first letter
        if description:
            description = description[0].upper() + description[1:]

        return description[:200]  # Limit length

    def _check_urgency(self, text: str) -> tuple[bool, float]:
        """Check if text contains urgency indicators."""
        max_score = 0.0

        for regex, weight in self._compiled_urgency_patterns:
            if regex.search(text):
                max_score = max(max_score, weight)

        return max_score >= 0.7, max_score

    def _calculate_priority(
        self,
        match_weight: float,
        urgency_score: float,
        action_type: ActionType,
    ) -> ActionItemPriority:
        """Calculate priority based on match weight and urgency."""
        # High urgency types
        high_urgency_types = {ActionType.APPROVE, ActionType.DECISION, ActionType.RESPOND}

        combined_score = match_weight * 0.6 + urgency_score * 0.4

        if action_type in high_urgency_types:
            combined_score += 0.1

        if combined_score >= 0.85 or urgency_score >= 0.9:
            return ActionItemPriority.CRITICAL
        elif combined_score >= 0.7:
            return ActionItemPriority.HIGH
        elif combined_score >= 0.5:
            return ActionItemPriority.MEDIUM
        else:
            return ActionItemPriority.LOW

    def _extract_deadline(
        self,
        sentence: str,
        full_text: str,
    ) -> tuple[Optional[datetime], Optional[str], bool]:
        """Extract deadline from text."""
        now = datetime.now(timezone.utc)

        for pattern, pattern_type in DEADLINE_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                match = re.search(pattern, full_text, re.IGNORECASE)

            if match:
                deadline_text = match.group(0)
                deadline = self._parse_deadline(match, pattern_type, now)

                if deadline:
                    is_explicit = pattern_type in ("date", "time", "weekday")
                    return deadline, deadline_text, is_explicit

        return None, None, False

    def _parse_deadline(
        self,
        match: re.Match[str],
        pattern_type: str,
        now: datetime,
    ) -> Optional[datetime]:
        """Parse deadline from regex match."""
        try:
            if pattern_type == "today":
                return now.replace(hour=17, minute=0, second=0, microsecond=0)

            elif pattern_type == "tomorrow":
                tomorrow = now + timedelta(days=1)
                return tomorrow.replace(hour=17, minute=0, second=0, microsecond=0)

            elif pattern_type == "weekday":
                weekday_text = match.group(1).lower()
                return self._next_weekday(now, weekday_text)

            elif pattern_type == "relative_end":
                period = match.group(1).lower()
                if period == "day":
                    return now.replace(hour=17, minute=0, second=0, microsecond=0)
                elif period == "week":
                    # End of current week (Friday 5pm)
                    days_until_friday = (4 - now.weekday()) % 7
                    if days_until_friday == 0 and now.hour >= 17:
                        days_until_friday = 7
                    friday = now + timedelta(days=days_until_friday)
                    return friday.replace(hour=17, minute=0, second=0, microsecond=0)
                elif period == "month":
                    # End of current month
                    next_month = now.replace(day=28) + timedelta(days=4)
                    end_of_month = next_month.replace(day=1) - timedelta(days=1)
                    return end_of_month.replace(hour=17, minute=0, second=0, microsecond=0)

            elif pattern_type == "eod_eow":
                period = match.group(1).lower()
                if "day" in period or "business" in period:
                    return now.replace(hour=17, minute=0, second=0, microsecond=0)
                elif "week" in period:
                    days_until_friday = (4 - now.weekday()) % 7
                    friday = now + timedelta(days=days_until_friday)
                    return friday.replace(hour=17, minute=0, second=0, microsecond=0)

            elif pattern_type == "within" or pattern_type == "in_time":
                amount = int(match.group(1))
                unit = match.group(2).lower()
                if "hour" in unit:
                    return now + timedelta(hours=amount)
                elif "day" in unit:
                    return now + timedelta(days=amount)
                elif "week" in unit:
                    return now + timedelta(weeks=amount)

            elif pattern_type == "date":
                date_str = match.group(1)
                # Try common date formats
                for fmt in ["%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%m/%d", "%m-%d"]:
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        if parsed.year == 1900:  # No year provided
                            parsed = parsed.replace(year=now.year)
                            if parsed < now:
                                parsed = parsed.replace(year=now.year + 1)
                        return parsed.replace(tzinfo=timezone.utc, hour=17)
                    except ValueError:
                        continue

            elif pattern_type == "relative_week":
                modifier = match.group(1).lower()
                day_text = match.group(2).lower()

                if day_text == "week":
                    # This week / next week
                    if modifier == "this":
                        days_until_friday = (4 - now.weekday()) % 7
                    else:
                        days_until_friday = (4 - now.weekday()) % 7 + 7
                    friday = now + timedelta(days=days_until_friday)
                    return friday.replace(hour=17, minute=0, second=0, microsecond=0)
                else:
                    # Specific weekday
                    target = self._next_weekday(now, day_text)
                    if modifier == "next" and target and target.date() == now.date():
                        target = target + timedelta(days=7)
                    return target

        except (ValueError, TypeError, IndexError):
            pass

        return None

    def _next_weekday(self, from_date: datetime, weekday_name: str) -> Optional[datetime]:
        """Get the next occurrence of a weekday."""
        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }

        target_weekday = weekday_map.get(weekday_name.lower())
        if target_weekday is None:
            return None

        days_ahead = (target_weekday - from_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next week if today

        target = from_date + timedelta(days=days_ahead)
        return target.replace(hour=17, minute=0, second=0, microsecond=0)

    def _detect_assignee(
        self,
        sentence: str,
        to_addresses: List[str],
        sender: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Detect who the action item is assigned to."""
        # Check for explicit mentions
        mention_patterns = [
            r"@(\w+)",  # @mentions
            r"\b(\w+),?\s+(?:please|can you|could you)",  # Name, please...
        ]

        for pattern in mention_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Try to find matching email
                for addr in to_addresses:
                    if name.lower() in addr.lower():
                        return addr, name

        # Check for "you" directed at recipients
        if re.search(r"\byou\b", sentence, re.IGNORECASE):
            # Likely directed at recipients
            if self.user_email and self.user_email in to_addresses:
                return self.user_email, self.user_name
            elif to_addresses:
                return to_addresses[0], None

        # Check for "I" (sender is responsible)
        if re.search(r"\bI('ll| will| need to)\b", sentence, re.IGNORECASE):
            return sender, None

        # Default: assign to recipients if it looks like a request
        if to_addresses and re.search(r"\b(please|can|could|need)\b", sentence, re.IGNORECASE):
            if self.user_email and self.user_email in to_addresses:
                return self.user_email, self.user_name
            return to_addresses[0], None

        return None, None

    def _get_context(self, sentences: List[str], current: str) -> str:
        """Get surrounding context for an action item."""
        try:
            idx = sentences.index(current)
            start = max(0, idx - 1)
            end = min(len(sentences), idx + 2)
            context_sentences = sentences[start:end]
            return " ".join(context_sentences)[:500]
        except ValueError:
            return current

    def _extract_tags(self, sentence: str, action_type: ActionType) -> List[str]:
        """Extract relevant tags from sentence."""
        tags = [action_type.value]

        # Check for common topic indicators
        topic_patterns = [
            (r"\b(project|feature|bug|issue)\b", "project"),
            (r"\b(document|doc|file|report)\b", "document"),
            (r"\b(meeting|call|sync)\b", "meeting"),
            (r"\b(code|pr|pull\s+request|commit)\b", "code"),
            (r"\b(invoice|payment|budget)\b", "finance"),
            (r"\b(hire|candidate|interview)\b", "hiring"),
            (r"\b(customer|client)\b", "customer"),
        ]

        for pattern, tag in topic_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                tags.append(tag)

        return tags

    async def extract_batch(
        self,
        emails: List[EmailMessage],
    ) -> List[ExtractionResult]:
        """Extract action items from multiple emails."""
        import asyncio

        tasks = [self.extract_action_items(email) for email in emails]
        return await asyncio.gather(*tasks)

    def mark_completed(self, action_item: ActionItem) -> ActionItem:
        """Mark an action item as completed."""
        action_item.status = ActionItemStatus.COMPLETED
        action_item.completed_at = datetime.now(timezone.utc)
        return action_item

    def get_pending_items(
        self,
        items: List[ActionItem],
        due_within_hours: Optional[int] = None,
    ) -> List[ActionItem]:
        """
        Get pending action items, optionally filtered by deadline.

        Args:
            items: List of action items to filter
            due_within_hours: Optional filter for items due within N hours

        Returns:
            Filtered list of pending action items
        """
        now = datetime.now(timezone.utc)
        pending = [
            item
            for item in items
            if item.status in (ActionItemStatus.PENDING, ActionItemStatus.IN_PROGRESS)
        ]

        if due_within_hours:
            cutoff = now + timedelta(hours=due_within_hours)
            pending = [item for item in pending if item.deadline and item.deadline <= cutoff]

        # Sort by deadline (items with deadlines first, then by priority)
        pending.sort(
            key=lambda x: (
                x.deadline is None,  # Items with deadlines first
                x.deadline or datetime.max.replace(tzinfo=timezone.utc),
                x.priority.value,
            )
        )

        return pending


# Convenience function
async def extract_action_items_quick(
    subject: str,
    body: str,
    sender: str = "",
) -> ExtractionResult:
    """
    Quick action item extraction without full email object.

    Args:
        subject: Email subject
        body: Email body text
        sender: Sender email address

    Returns:
        ExtractionResult with extracted action items
    """

    class SimpleEmail:
        def __init__(self, subject: str, body: str, sender: str):
            self.id = f"quick_{hash((subject, body, sender))}"
            self.subject = subject
            self.body_text = body
            self.from_address = sender
            self.to_addresses: List[str] = []

    email = SimpleEmail(subject, body, sender)
    extractor = ActionItemExtractor()
    return await extractor.extract_action_items(email)  # type: ignore


__all__ = [
    "ActionItemExtractor",
    "ActionItem",
    "ExtractionResult",
    "ActionItemPriority",
    "ActionItemStatus",
    "ActionType",
    "extract_action_items_quick",
]
