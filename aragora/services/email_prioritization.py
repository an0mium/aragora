"""
Email Prioritization Service.

Intelligent email inbox management using Aragora's multi-agent debate system.
Implements a 3-tier scoring architecture:

- Tier 1 (Fast, <200ms): Rule-based scoring using sender reputation and keywords
- Tier 2 (Medium, <500ms): Single-agent lightweight analysis for ambiguous cases
- Tier 3 (Async): Full multi-agent debate for complex prioritization decisions

Features:
- Cross-channel context integration (Slack, Google Drive, calendar)
- Decision receipts explaining prioritization rationale
- Continuous learning from user actions
- VIP sender management
- Time-sensitive deadline detection

Usage:
    from aragora.services.email_prioritization import EmailPrioritizer, EmailPriorityRequest

    prioritizer = EmailPrioritizer(
        gmail_connector=gmail_connector,
        knowledge_mound=mound,
    )

    # Score a single email
    result = await prioritizer.score_email(email_message)
    print(f"Priority: {result.priority} (confidence: {result.confidence})")
    print(f"Rationale: {result.rationale}")

    # Batch process inbox
    ranked_emails = await prioritizer.rank_inbox(emails)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from aragora.connectors.enterprise.communication.gmail import GmailConnector
    from aragora.connectors.enterprise.communication.models import EmailMessage
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class EmailPriority(Enum):
    """Email priority levels."""

    CRITICAL = 1  # Immediate attention required
    HIGH = 2  # Important, respond today
    MEDIUM = 3  # Standard priority
    LOW = 4  # Can wait, review when time allows
    DEFER = 5  # Archive or auto-file


class ScoringTier(Enum):
    """Which scoring tier was used."""

    TIER_1_RULES = "tier_1_rules"
    TIER_2_LIGHTWEIGHT = "tier_2_lightweight"
    TIER_3_DEBATE = "tier_3_debate"


@dataclass
class SenderProfile:
    """Profile of an email sender for reputation scoring."""

    email: str
    domain: str
    is_vip: bool = False
    is_internal: bool = False
    response_rate: float = 0.0  # How often user responds to this sender
    avg_response_time_hours: float = 24.0  # Average response time
    last_interaction: Optional[datetime] = None
    total_emails_received: int = 0
    total_emails_responded: int = 0
    tags: Set[str] = field(default_factory=set)

    @property
    def reputation_score(self) -> float:
        """Calculate sender reputation score (0-1)."""
        score = 0.5  # Baseline

        if self.is_vip:
            score += 0.3
        if self.is_internal:
            score += 0.1
        if self.response_rate > 0.8:
            score += 0.1
        elif self.response_rate < 0.1:
            score -= 0.2

        # Recent interaction bonus
        if self.last_interaction:
            days_since = (datetime.now() - self.last_interaction).days
            if days_since < 7:
                score += 0.1
            elif days_since > 90:
                score -= 0.1

        return max(0.0, min(1.0, score))


@dataclass
class EmailPriorityResult:
    """Result of email prioritization."""

    email_id: str
    priority: EmailPriority
    confidence: float
    tier_used: ScoringTier
    rationale: str

    # Detailed scoring breakdown
    sender_score: float = 0.0
    content_urgency_score: float = 0.0
    context_relevance_score: float = 0.0
    time_sensitivity_score: float = 0.0

    # Cross-channel signals
    slack_activity_boost: float = 0.0
    calendar_relevance_boost: float = 0.0
    drive_activity_boost: float = 0.0

    # Debate metadata (if Tier 3 was used)
    debate_id: Optional[str] = None
    agent_dissent: Optional[Dict[str, Any]] = None

    # Recommended actions
    suggested_labels: List[str] = field(default_factory=list)
    suggested_response: Optional[str] = None
    auto_archive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "email_id": self.email_id,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "confidence": self.confidence,
            "tier_used": self.tier_used.value,
            "rationale": self.rationale,
            "scores": {
                "sender": self.sender_score,
                "content_urgency": self.content_urgency_score,
                "context_relevance": self.context_relevance_score,
                "time_sensitivity": self.time_sensitivity_score,
            },
            "cross_channel_boosts": {
                "slack": self.slack_activity_boost,
                "calendar": self.calendar_relevance_boost,
                "drive": self.drive_activity_boost,
            },
            "debate_id": self.debate_id,
            "suggested_labels": self.suggested_labels,
            "auto_archive": self.auto_archive,
        }


@dataclass
class EmailPrioritizationConfig:
    """Configuration for email prioritization."""

    # Tier thresholds
    tier_1_confidence_threshold: float = 0.7  # Below this, escalate to Tier 2
    tier_2_confidence_threshold: float = 0.6  # Below this, escalate to Tier 3

    # VIP settings
    vip_domains: Set[str] = field(default_factory=set)
    vip_addresses: Set[str] = field(default_factory=set)
    internal_domains: Set[str] = field(default_factory=set)

    # Auto-archive patterns
    auto_archive_senders: Set[str] = field(default_factory=set)
    newsletter_patterns: List[str] = field(
        default_factory=lambda: [
            r"unsubscribe",
            r"email preferences",
            r"opt.out",
            r"newsletter",
            r"no.?reply",
        ]
    )

    # Urgency keywords
    urgent_keywords: List[str] = field(
        default_factory=lambda: [
            "urgent",
            "asap",
            "immediately",
            "critical",
            "emergency",
            "deadline",
            "today",
            "eod",
            "end of day",
            "time-sensitive",
        ]
    )

    # Cross-channel integration
    enable_slack_signals: bool = True
    enable_calendar_signals: bool = True
    enable_drive_signals: bool = True

    # Debate settings
    debate_agent_count: int = 3
    debate_timeout_seconds: float = 30.0


# Urgency detection patterns
DEADLINE_PATTERNS = [
    r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"by\s+(\d{1,2}[/-]\d{1,2})",
    r"due\s+(today|tomorrow|this\s+week)",
    r"deadline[:\s]+(\w+)",
    r"need\s+by\s+",
    r"respond\s+by\s+",
]


class EmailPrioritizer:
    """
    Intelligent email prioritization using multi-agent debate.

    Implements 3-tier scoring for optimal latency/accuracy tradeoff:
    - Tier 1: Rule-based (fast, high-confidence cases)
    - Tier 2: Lightweight single-agent (ambiguous cases)
    - Tier 3: Full debate (complex decisions)
    """

    def __init__(
        self,
        gmail_connector: Optional["GmailConnector"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        config: Optional[EmailPrioritizationConfig] = None,
    ):
        """
        Initialize email prioritizer.

        Args:
            gmail_connector: Gmail connector for fetching emails
            knowledge_mound: Knowledge Mound for context and learning
            config: Prioritization configuration
        """
        self.gmail = gmail_connector
        self.mound = knowledge_mound
        self.config = config or EmailPrioritizationConfig()

        # Sender profile cache
        self._sender_profiles: Dict[str, SenderProfile] = {}

        # Compile patterns
        self._newsletter_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.newsletter_patterns
        ]
        self._urgent_patterns = [
            re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in self.config.urgent_keywords
        ]
        self._deadline_patterns = [re.compile(p, re.IGNORECASE) for p in DEADLINE_PATTERNS]

    async def score_email(
        self,
        email: "EmailMessage",
        force_tier: Optional[ScoringTier] = None,
    ) -> EmailPriorityResult:
        """
        Score a single email for priority.

        Uses 3-tier system:
        1. Tier 1: Fast rule-based scoring
        2. If low confidence, escalate to Tier 2 (lightweight agent)
        3. If still ambiguous, escalate to Tier 3 (full debate)

        Args:
            email: Email message to score
            force_tier: Force a specific tier (for testing)

        Returns:
            EmailPriorityResult with priority and rationale
        """
        # Get or create sender profile
        sender_profile = await self._get_sender_profile(email.from_address)

        # Tier 1: Rule-based scoring
        if force_tier is None or force_tier == ScoringTier.TIER_1_RULES:
            result = await self._tier_1_score(email, sender_profile)

            if result.confidence >= self.config.tier_1_confidence_threshold:
                return result

            if force_tier == ScoringTier.TIER_1_RULES:
                return result

        # Tier 2: Lightweight agent scoring
        if force_tier is None or force_tier == ScoringTier.TIER_2_LIGHTWEIGHT:
            result = await self._tier_2_score(email, sender_profile)

            if result.confidence >= self.config.tier_2_confidence_threshold:
                return result

            if force_tier == ScoringTier.TIER_2_LIGHTWEIGHT:
                return result

        # Tier 3: Full multi-agent debate
        result = await self._tier_3_debate(email, sender_profile)
        return result

    async def rank_inbox(
        self,
        emails: List["EmailMessage"],
        limit: Optional[int] = None,
    ) -> List[EmailPriorityResult]:
        """
        Rank a list of emails by priority.

        Args:
            emails: List of emails to rank
            limit: Optional limit on results

        Returns:
            List of EmailPriorityResult sorted by priority
        """
        # Score all emails concurrently
        tasks = [self.score_email(email) for email in emails]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        scored_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Failed to score email: {result}")
            else:
                scored_results.append(result)

        # Sort by priority (lower = more important), then by confidence
        scored_results.sort(key=lambda r: (r.priority.value, -r.confidence))  # type: ignore[union-attr]

        if limit:
            scored_results = scored_results[:limit]

        return scored_results  # type: ignore[return-value]

    async def _get_sender_profile(self, email_address: str) -> SenderProfile:
        """Get or create sender profile from cache/mound."""
        if email_address in self._sender_profiles:
            return self._sender_profiles[email_address]

        # Parse domain
        domain = email_address.split("@")[-1] if "@" in email_address else ""

        # Create profile
        profile = SenderProfile(
            email=email_address,
            domain=domain,
            is_vip=(
                email_address.lower() in {a.lower() for a in self.config.vip_addresses}
                or domain.lower() in {d.lower() for d in self.config.vip_domains}
            ),
            is_internal=domain.lower() in {d.lower() for d in self.config.internal_domains},
        )

        # Try to load history from knowledge mound
        if self.mound:
            try:
                history = await self._load_sender_history(email_address)
                if history:
                    profile.response_rate = history.get("response_rate", 0.0)
                    profile.avg_response_time_hours = history.get("avg_response_time", 24.0)
                    profile.total_emails_received = history.get("total_received", 0)
                    profile.total_emails_responded = history.get("total_responded", 0)
                    if history.get("last_interaction"):
                        profile.last_interaction = datetime.fromisoformat(
                            history["last_interaction"]
                        )
            except Exception as e:
                logger.warning(f"Failed to load sender history: {e}")

        self._sender_profiles[email_address] = profile
        return profile

    async def _load_sender_history(self, email_address: str) -> Optional[Dict[str, Any]]:
        """Load sender interaction history from Knowledge Mound."""
        if not self.mound:
            return None

        try:
            # Query for sender history node
            results = await self.mound.query(
                query=f"sender_history:{email_address}",
                limit=1,
            )
            if results and hasattr(results, "items") and results.items:
                return results.items[0].metadata
        except Exception as e:
            logger.debug(f"No sender history found: {e}")

        return None

    async def _tier_1_score(
        self,
        email: "EmailMessage",
        sender: SenderProfile,
    ) -> EmailPriorityResult:
        """
        Tier 1: Fast rule-based scoring (<200ms target).

        Uses:
        - Sender reputation
        - Keyword detection
        - Pattern matching
        - Gmail labels
        """
        scores = {
            "sender": sender.reputation_score,
            "content_urgency": 0.0,
            "context_relevance": 0.5,  # Default neutral
            "time_sensitivity": 0.0,
        }
        rationale_parts = []
        confidence = 0.8  # Start high for rule-based

        # Check for auto-archive patterns (newsletters, bulk)
        is_newsletter = self._detect_newsletter(email)
        if is_newsletter:
            return EmailPriorityResult(
                email_id=email.id,
                priority=EmailPriority.DEFER,
                confidence=0.9,
                tier_used=ScoringTier.TIER_1_RULES,
                rationale="Detected as newsletter/bulk email",
                sender_score=scores["sender"],
                auto_archive=True,
                suggested_labels=["Newsletter"],
            )

        # VIP sender check
        if sender.is_vip:
            scores["sender"] = min(1.0, scores["sender"] + 0.3)
            rationale_parts.append("VIP sender")

        # Internal sender check
        if sender.is_internal:
            scores["sender"] = min(1.0, scores["sender"] + 0.1)
            rationale_parts.append("Internal sender")

        # Gmail label signals
        if email.is_important:
            scores["content_urgency"] += 0.2
            rationale_parts.append("Marked important by Gmail")

        if email.is_starred:
            scores["content_urgency"] += 0.3
            rationale_parts.append("Starred")

        # Urgency keyword detection
        text = f"{email.subject} {email.body_text or ''}"
        urgency_matches = sum(1 for pattern in self._urgent_patterns if pattern.search(text))
        if urgency_matches > 0:
            scores["content_urgency"] += min(0.4, urgency_matches * 0.15)
            rationale_parts.append(f"Urgency keywords detected ({urgency_matches})")

        # Deadline detection
        deadline_match = None
        for pattern in self._deadline_patterns:
            match = pattern.search(text)
            if match:
                deadline_match = match.group(0)
                scores["time_sensitivity"] += 0.3
                rationale_parts.append(f"Deadline detected: {deadline_match}")
                break

        # Reply-needed detection
        if self._detect_reply_needed(email):
            scores["content_urgency"] += 0.2
            rationale_parts.append("Reply expected")

        # Calculate weighted priority score
        weights = {
            "sender": 0.3,
            "content_urgency": 0.35,
            "context_relevance": 0.2,
            "time_sensitivity": 0.15,
        }

        priority_score = sum(scores[k] * weights[k] for k in weights)

        # Map score to priority
        if priority_score >= 0.75:
            priority = EmailPriority.CRITICAL
        elif priority_score >= 0.55:
            priority = EmailPriority.HIGH
        elif priority_score >= 0.35:
            priority = EmailPriority.MEDIUM
        elif priority_score >= 0.2:
            priority = EmailPriority.LOW
        else:
            priority = EmailPriority.DEFER

        # Adjust confidence based on signal strength
        if not rationale_parts:
            confidence = 0.5  # Low confidence without clear signals
        elif len(rationale_parts) >= 3:
            confidence = 0.85  # High confidence with multiple signals

        rationale = "; ".join(rationale_parts) if rationale_parts else "No strong priority signals"

        return EmailPriorityResult(
            email_id=email.id,
            priority=priority,
            confidence=confidence,
            tier_used=ScoringTier.TIER_1_RULES,
            rationale=rationale,
            sender_score=scores["sender"],
            content_urgency_score=scores["content_urgency"],
            context_relevance_score=scores["context_relevance"],
            time_sensitivity_score=scores["time_sensitivity"],
        )

    async def _tier_2_score(
        self,
        email: "EmailMessage",
        sender: SenderProfile,
    ) -> EmailPriorityResult:
        """
        Tier 2: Lightweight single-agent scoring (<500ms target).

        Uses a small, fast model to analyze content when
        rule-based scoring is ambiguous.
        """
        # Start with Tier 1 scores as baseline
        tier_1_result = await self._tier_1_score(email, sender)

        # Build prompt for lightweight analysis
        prompt = f"""Analyze this email and rate its priority from 1-5:
1 = Critical (immediate attention)
2 = High (respond today)
3 = Medium (standard)
4 = Low (can wait)
5 = Defer (archive)

Subject: {email.subject}
From: {email.from_address}
Snippet: {email.snippet[:300] if email.snippet else email.body_text[:300] if email.body_text else ''}

Context:
- Sender reputation: {'VIP' if sender.is_vip else 'Known' if sender.reputation_score > 0.5 else 'Unknown'}
- Gmail flags: {'Important' if email.is_important else ''} {'Starred' if email.is_starred else ''}

Output format: PRIORITY: [1-5], CONFIDENCE: [0-1], REASON: [brief explanation]"""

        try:
            # Use lightweight model for quick analysis
            from aragora.core.model_router import get_model_router

            router = get_model_router()
            response = await router.generate(
                prompt=prompt,
                model_preference="fast",  # Request lightweight model
                max_tokens=100,
            )

            # Parse response
            text = response.content if hasattr(response, "content") else str(response)

            # Extract priority
            priority_match = re.search(r"PRIORITY:\s*(\d)", text)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
            reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)

            if priority_match:
                priority_val = int(priority_match.group(1))
                priority = EmailPriority(min(5, max(1, priority_val)))
            else:
                priority = tier_1_result.priority

            if confidence_match:
                confidence = float(confidence_match.group(1))
            else:
                confidence = 0.6

            reason = reason_match.group(1).strip() if reason_match else tier_1_result.rationale

            return EmailPriorityResult(
                email_id=email.id,
                priority=priority,
                confidence=confidence,
                tier_used=ScoringTier.TIER_2_LIGHTWEIGHT,
                rationale=reason,
                sender_score=tier_1_result.sender_score,
                content_urgency_score=tier_1_result.content_urgency_score,
                context_relevance_score=tier_1_result.context_relevance_score,
                time_sensitivity_score=tier_1_result.time_sensitivity_score,
            )

        except Exception as e:
            logger.warning(f"Tier 2 scoring failed, using Tier 1 result: {e}")
            tier_1_result.tier_used = ScoringTier.TIER_2_LIGHTWEIGHT
            tier_1_result.rationale += " (Tier 2 fallback to rules)"
            return tier_1_result

    async def _tier_3_debate(
        self,
        email: "EmailMessage",
        sender: SenderProfile,
    ) -> EmailPriorityResult:
        """
        Tier 3: Full multi-agent debate for complex decisions.

        Runs asynchronously with multiple specialized agents:
        - SenderReputationAgent: Analyzes sender importance
        - ContentUrgencyAgent: Detects urgency and deadlines
        - ContextRelevanceAgent: Cross-references with knowledge mound
        """
        # Start with Tier 2 scores as baseline
        tier_2_result = await self._tier_2_score(email, sender)

        try:
            from aragora.debate.arena import DebateArena

            # Build debate question
            question = f"""Prioritize this email (1=Critical to 5=Defer):

Subject: {email.subject}
From: {email.from_address} ({"VIP" if sender.is_vip else "reputation: " + f"{sender.reputation_score:.2f}"})
Body preview: {(email.body_text or email.body_html or "")[:500]}

Consider:
1. Who is the sender and their relationship to the user?
2. Is there a deadline or time-sensitive request?
3. What action (if any) is requested?
4. How does this relate to user's current priorities?

Provide: PRIORITY (1-5), CONFIDENCE (0-1), and RATIONALE."""

            # Create arena and run debate
            arena = DebateArena(
                agents=["anthropic-api", "openai-api", "gemini"],  # Heterogeneous team
            )

            result = await arena.debate(
                question=question,
                rounds=2,
                timeout=self.config.debate_timeout_seconds,
            )

            # Parse consensus result
            if result and hasattr(result, "final_answer"):
                answer = result.final_answer

                # Extract priority from debate result
                priority_match = re.search(r"PRIORITY[:\s]*(\d)", answer, re.IGNORECASE)
                confidence_match = re.search(r"CONFIDENCE[:\s]*([\d.]+)", answer, re.IGNORECASE)

                if priority_match:
                    priority_val = int(priority_match.group(1))
                    priority = EmailPriority(min(5, max(1, priority_val)))
                else:
                    priority = tier_2_result.priority

                confidence = (
                    float(confidence_match.group(1)) if confidence_match else result.confidence
                )

                return EmailPriorityResult(
                    email_id=email.id,
                    priority=priority,
                    confidence=confidence,
                    tier_used=ScoringTier.TIER_3_DEBATE,
                    rationale=answer[:500],  # Truncate long rationales
                    sender_score=tier_2_result.sender_score,
                    content_urgency_score=tier_2_result.content_urgency_score,
                    context_relevance_score=tier_2_result.context_relevance_score,
                    time_sensitivity_score=tier_2_result.time_sensitivity_score,
                    debate_id=result.debate_id if hasattr(result, "debate_id") else None,
                    agent_dissent=result.dissent if hasattr(result, "dissent") else None,
                )

        except Exception as e:
            logger.warning(f"Tier 3 debate failed, using Tier 2 result: {e}")
            tier_2_result.tier_used = ScoringTier.TIER_3_DEBATE
            tier_2_result.rationale += " (Tier 3 fallback to lightweight)"

        return tier_2_result

    def _detect_newsletter(self, email: "EmailMessage") -> bool:
        """Detect if email is a newsletter/bulk mail."""
        # Check sender in auto-archive list
        if email.from_address.lower() in {a.lower() for a in self.config.auto_archive_senders}:
            return True

        # Check for unsubscribe link in headers
        if "list-unsubscribe" in email.headers:
            return True

        # Check content patterns
        text = f"{email.subject} {email.body_text or ''} {email.body_html or ''}"
        for pattern in self._newsletter_patterns:
            if pattern.search(text):
                return True

        # Check for no-reply sender
        if "noreply" in email.from_address.lower() or "no-reply" in email.from_address.lower():
            return True

        return False

    def _detect_reply_needed(self, email: "EmailMessage") -> bool:
        """Detect if email expects a reply."""
        text = f"{email.subject} {email.body_text or ''}"

        reply_patterns = [
            r"\?$",  # Ends with question mark
            r"please\s+(respond|reply|let\s+me\s+know)",
            r"(can|could|would)\s+you",
            r"what\s+do\s+you\s+think",
            r"your\s+(thoughts|input|feedback)",
            r"looking\s+forward\s+to\s+(hearing|your)",
        ]

        for pattern in reply_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    async def record_user_action(
        self,
        email_id: str,
        action: str,
        email: Optional["EmailMessage"] = None,
        user_id: str = "default",
        response_time_minutes: Optional[int] = None,
    ) -> None:
        """
        Record user action for learning.

        Actions: read, opened, archived, deleted, replied, starred, important

        Args:
            email_id: Email identifier
            action: Type of action performed
            email: Optional email message for context
            user_id: User identifier for multi-user support
            response_time_minutes: Time taken to respond (for reply actions)
        """
        # Record in sender history service if available
        if email:
            try:
                from aragora.services.sender_history import SenderHistoryService

                # Try to get sender history service from registry
                try:
                    from aragora.services import get_service, has_service

                    if has_service(SenderHistoryService):
                        history_service = get_service(SenderHistoryService)
                    else:
                        # Create a temporary service instance
                        history_service = SenderHistoryService()
                        await history_service.initialize()
                except (ImportError, KeyError, AttributeError) as e:
                    logger.debug(f"Service registry unavailable, creating temporary instance: {e}")
                    history_service = SenderHistoryService()
                    await history_service.initialize()
                except Exception as e:
                    logger.warning(f"Unexpected error getting history service: {e}")
                    history_service = SenderHistoryService()
                    await history_service.initialize()

                # Map actions to interaction types
                interaction_action = action
                if action == "read":
                    interaction_action = "opened"
                elif action == "starred" or action == "important":
                    interaction_action = "starred"

                await history_service.record_interaction(
                    user_id=user_id,
                    sender_email=email.from_address,
                    action=interaction_action,
                    email_id=email_id,
                    response_time_minutes=response_time_minutes,
                    metadata={
                        "subject": email.subject[:100] if email.subject else None,
                        "labels": email.labels,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to record sender history: {e}")

        # Store action in knowledge mound for learning
        if self.mound:
            try:
                await self.mound.store(  # type: ignore[misc,call-arg]
                    content=f"User {action} email {email_id}",
                    metadata={
                        "type": "email_action",
                        "email_id": email_id,
                        "action": action,
                        "timestamp": datetime.now().isoformat(),
                        "sender": email.from_address if email else None,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to store in knowledge mound: {e}")

        # Update sender profile if we have the email
        if email:
            try:
                profile = await self._get_sender_profile(email.from_address)
                profile.total_emails_received += 1

                if action == "replied":
                    profile.total_emails_responded += 1
                    profile.response_rate = (
                        profile.total_emails_responded / profile.total_emails_received
                    )

                profile.last_interaction = datetime.now()
            except Exception as e:
                logger.debug(f"Failed to update sender profile: {e}")


# Convenience function for quick access
async def prioritize_inbox(
    gmail_connector: "GmailConnector",
    knowledge_mound: Optional["KnowledgeMound"] = None,
    config: Optional[EmailPrioritizationConfig] = None,
    limit: int = 50,
) -> List[EmailPriorityResult]:
    """
    Quick function to prioritize an inbox.

    Args:
        gmail_connector: Authenticated Gmail connector
        knowledge_mound: Optional KM for context
        config: Optional prioritization config
        limit: Max emails to process

    Returns:
        List of prioritized emails
    """
    prioritizer = EmailPrioritizer(
        gmail_connector=gmail_connector,
        knowledge_mound=knowledge_mound,
        config=config,
    )

    # Fetch recent emails
    emails: list[Any] = []
    async for item in gmail_connector.sync():  # type: ignore[attr-defined]
        if len(emails) >= limit:
            break
        # Convert SyncItem to EmailMessage if needed
        if hasattr(item, "raw_data") and "message" in item.raw_data:
            emails.append(item.raw_data["message"])
        else:
            emails.append(item)

    return await prioritizer.rank_inbox(emails)
