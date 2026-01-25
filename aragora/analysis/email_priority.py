"""
Email Priority Analyzer.

Uses AragoraRLM (routes to TRUE RLM when available) and ContinuumMemory
to score email importance based on user-specific learned preferences.

TRUE RLM (when official library installed):
- Model writes code to programmatically analyze email content
- Context stored as REPL variables, not in prompt

Compression fallback (when official library not installed):
- Uses hierarchical summarization for long content
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

# Check for AragoraRLM (routes to TRUE RLM when available)
try:
    from aragora.rlm.bridge import AragoraRLM, HAS_OFFICIAL_RLM

    HAS_ARAGORA_RLM = True
except ImportError:
    HAS_ARAGORA_RLM = False
    HAS_OFFICIAL_RLM = False
    AragoraRLM: Optional[Type[Any]] = None  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass
class EmailPriorityScore:
    """Result of email priority scoring."""

    email_id: str
    score: float  # 0.0 - 1.0
    reason: str  # AI-generated explanation
    factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "email_id": self.email_id,
            "score": self.score,
            "reason": self.reason,
            "factors": self.factors,
        }


@dataclass
class UserEmailPreferences:
    """Learned user preferences for email prioritization."""

    user_id: str
    important_senders: List[str] = field(default_factory=list)
    important_domains: List[str] = field(default_factory=list)
    important_keywords: List[str] = field(default_factory=list)
    low_priority_senders: List[str] = field(default_factory=list)
    low_priority_keywords: List[str] = field(default_factory=list)
    interaction_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "important_senders": self.important_senders,
            "important_domains": self.important_domains,
            "important_keywords": self.important_keywords,
            "low_priority_senders": self.low_priority_senders,
            "low_priority_keywords": self.low_priority_keywords,
            "interaction_weights": self.interaction_weights,
        }


class EmailPriorityAnalyzer:
    """
    Analyze email importance using RLM and user memory.

    Combines multiple signals:
    - Sender importance (based on past interactions)
    - Content relevance (using RLM analysis)
    - Temporal urgency (keywords, deadlines)
    - Thread engagement (reply frequency)
    - Gmail signals (starred, important labels)
    """

    def __init__(
        self,
        user_id: str,
        memory: Optional[Any] = None,
        rlm: Optional[Any] = None,
    ):
        """
        Initialize analyzer.

        Args:
            user_id: User ID for preference lookups
            memory: ContinuumMemory instance (optional)
            rlm: StreamingRLMQuery instance (optional)
        """
        self.user_id = user_id
        self._memory = memory
        self._rlm = rlm
        self._preferences: Optional[UserEmailPreferences] = None

    async def _get_memory(self) -> Any:
        """Get or create ContinuumMemory instance."""
        if self._memory is None:
            try:
                from aragora.memory.continuum import ContinuumMemory

                self._memory = ContinuumMemory(user_id=self.user_id)  # type: ignore[call-arg]
                await self._memory.initialize()  # type: ignore[attr-defined]
            except ImportError:
                logger.debug("[EmailPriority] ContinuumMemory not available")
                return None
        return self._memory

    async def _get_rlm(self) -> Any:
        """Get or create AragoraRLM instance (routes to TRUE RLM when available)."""
        if self._rlm is None:
            if not HAS_ARAGORA_RLM or AragoraRLM is None:
                logger.debug("[EmailPriority] AragoraRLM not available")
                return None

            try:
                self._rlm = AragoraRLM()
                if HAS_OFFICIAL_RLM:
                    logger.info(
                        "[EmailPriority] TRUE RLM initialized "
                        "(REPL-based, model writes code to examine content)"
                    )
                else:
                    logger.info(
                        "[EmailPriority] AragoraRLM initialized "
                        "(will use compression fallback since official RLM not installed)"
                    )
            except Exception as e:
                logger.debug(f"[EmailPriority] Failed to initialize AragoraRLM: {e}")
                return None
        return self._rlm

    async def get_user_preferences(self) -> UserEmailPreferences:
        """Load user preferences from memory."""
        if self._preferences:
            return self._preferences

        self._preferences = UserEmailPreferences(user_id=self.user_id)

        memory = await self._get_memory()
        if not memory:
            return self._preferences

        try:
            # Query fast tier for recent preferences
            results = await memory.query(
                f"email preferences for user {self.user_id}",
                tiers=["fast", "medium"],
                limit=10,
            )

            # Parse preference signals from memory
            for result in results:
                content = result.get("content", "")

                # Extract senders user interacted with positively
                if "replied to" in content.lower() or "starred email from" in content.lower():
                    sender = self._extract_sender(content)
                    if sender and sender not in self._preferences.important_senders:
                        self._preferences.important_senders.append(sender)

                # Extract keywords from important topics
                if "important topic" in content.lower():
                    keywords = self._extract_keywords(content)
                    for kw in keywords:
                        if kw not in self._preferences.important_keywords:
                            self._preferences.important_keywords.append(kw)

        except Exception as e:
            logger.warning(f"[EmailPriority] Failed to load preferences: {e}")

        return self._preferences

    def _extract_sender(self, content: str) -> Optional[str]:
        """Extract sender email from memory content."""
        import re

        match = re.search(r"[\w.-]+@[\w.-]+\.\w+", content)
        return match.group(0) if match else None

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content."""
        # Simple keyword extraction - could use NLP in production
        words = content.lower().split()
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "about",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
        }
        return [w for w in words if len(w) > 3 and w not in stopwords][:5]

    async def score_email(
        self,
        email_id: str,
        subject: str,
        from_address: str,
        snippet: str,
        body_text: str = "",
        labels: Optional[List[str]] = None,
        is_read: bool = False,
        is_starred: bool = False,
        thread_count: int = 1,
    ) -> EmailPriorityScore:
        """
        Compute importance score for an email.

        Args:
            email_id: Email ID
            subject: Email subject
            from_address: Sender email
            snippet: Email preview snippet
            body_text: Full email body (optional)
            labels: Gmail labels
            is_read: Whether email is read
            is_starred: Whether email is starred
            thread_count: Number of messages in thread

        Returns:
            EmailPriorityScore with 0.0-1.0 score and explanation
        """
        labels = labels or []
        factors: Dict[str, float] = {}

        # Load user preferences
        prefs = await self.get_user_preferences()

        # 1. Sender importance (0.0 - 0.3)
        sender_score = self._score_sender(from_address, prefs)
        factors["sender"] = sender_score

        # 2. Gmail signals (0.0 - 0.2)
        gmail_score = self._score_gmail_signals(labels, is_starred)
        factors["gmail_signals"] = gmail_score

        # 3. Urgency signals (0.0 - 0.2)
        urgency_score = self._score_urgency(subject, snippet)
        factors["urgency"] = urgency_score

        # 4. Content relevance (0.0 - 0.2)
        content_score = await self._score_content(subject, snippet, body_text, prefs)
        factors["content"] = content_score

        # 5. Thread engagement (0.0 - 0.1)
        thread_score = self._score_thread(thread_count, is_read)
        factors["thread"] = thread_score

        # Compute weighted total
        total_score = (
            sender_score * 0.30
            + gmail_score * 0.20
            + urgency_score * 0.20
            + content_score * 0.20
            + thread_score * 0.10
        )

        # Clamp to 0.0-1.0
        total_score = max(0.0, min(1.0, total_score))

        # Generate explanation
        reason = self._generate_reason(factors, from_address, subject)

        return EmailPriorityScore(
            email_id=email_id,
            score=total_score,
            reason=reason,
            factors=factors,
        )

    def _score_sender(self, from_address: str, prefs: UserEmailPreferences) -> float:
        """Score based on sender importance."""
        score = 0.5  # Neutral baseline

        # Known important sender
        if from_address.lower() in [s.lower() for s in prefs.important_senders]:
            score = 1.0

        # Known low priority sender
        elif from_address.lower() in [s.lower() for s in prefs.low_priority_senders]:
            score = 0.2

        # Important domain
        else:
            domain = from_address.split("@")[-1].lower() if "@" in from_address else ""
            if domain in [d.lower() for d in prefs.important_domains]:
                score = 0.8

            # Check for work/corporate domains
            elif any(x in domain for x in [".com", ".io", ".ai", ".co"]):
                if not any(x in domain for x in ["gmail", "yahoo", "hotmail", "outlook"]):
                    score = 0.6

        return score

    def _score_gmail_signals(self, labels: List[str], is_starred: bool) -> float:
        """Score based on Gmail labels and flags."""
        score = 0.5

        # Starred emails are important
        if is_starred:
            score = 1.0

        # Gmail's IMPORTANT label
        elif "IMPORTANT" in labels:
            score = 0.9

        # Primary inbox
        elif "CATEGORY_PRIMARY" in labels:
            score = 0.7

        # Promotions/Social/Forums
        elif any(
            label in labels
            for label in ["CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"]
        ):
            score = 0.3

        # Updates
        elif "CATEGORY_UPDATES" in labels:
            score = 0.5

        return score

    def _score_urgency(self, subject: str, snippet: str) -> float:
        """Score based on urgency signals in content."""
        text = f"{subject} {snippet}".lower()
        score = 0.5

        # High urgency keywords
        high_urgency = [
            "urgent",
            "asap",
            "immediate",
            "deadline",
            "today",
            "eod",
            "emergency",
            "critical",
            "time sensitive",
            "action required",
        ]
        if any(kw in text for kw in high_urgency):
            score = 1.0

        # Medium urgency
        elif any(
            kw in text
            for kw in [
                "tomorrow",
                "soon",
                "important",
                "please review",
                "waiting for",
                "follow up",
                "reminder",
            ]
        ):
            score = 0.7

        # Low urgency (newsletter-like)
        elif any(
            kw in text
            for kw in [
                "unsubscribe",
                "newsletter",
                "weekly digest",
                "monthly update",
                "no reply needed",
            ]
        ):
            score = 0.2

        return score

    async def _score_content(
        self,
        subject: str,
        snippet: str,
        body_text: str,
        prefs: UserEmailPreferences,
    ) -> float:
        """Score based on content relevance using RLM."""
        score = 0.5
        text = f"{subject} {snippet}".lower()

        # Check for important keywords from preferences
        keyword_matches = sum(1 for kw in prefs.important_keywords if kw.lower() in text)
        if keyword_matches > 0:
            score = min(1.0, 0.5 + keyword_matches * 0.2)

        # Check for low priority keywords
        low_priority_matches = sum(1 for kw in prefs.low_priority_keywords if kw.lower() in text)
        if low_priority_matches > 0:
            score = max(0.1, score - low_priority_matches * 0.2)

        # Use AragoraRLM for deeper analysis if available and body is long
        # Routes to TRUE RLM (REPL-based) when available, compression fallback otherwise
        rlm = await self._get_rlm()
        if rlm and body_text and len(body_text) > 200:
            try:
                # Analyze relevance using AragoraRLM
                analysis_result = await rlm.compress_and_query(
                    query=(
                        "Rate this email's importance on a scale of 0-10. "
                        "Consider if it requires action, contains time-sensitive info, "
                        "or is from an important context. Reply with just the number."
                    ),
                    content=body_text[:2000],
                    source_type="email",
                )

                if analysis_result and analysis_result.answer:
                    # Log which approach was used
                    if analysis_result.used_true_rlm:
                        logger.debug("[EmailPriority] Used TRUE RLM for content analysis")
                    elif analysis_result.used_compression_fallback:
                        logger.debug(
                            "[EmailPriority] Used compression fallback for content analysis"
                        )

                    try:
                        # Extract numeric score from response
                        answer = analysis_result.answer.strip()
                        # Try to find a number in the response
                        import re

                        match = re.search(r"\b(\d+(?:\.\d+)?)\b", answer)
                        if match:
                            rlm_score = float(match.group(1)) / 10.0
                            rlm_score = max(0.0, min(1.0, rlm_score))  # Clamp
                            # Blend with keyword-based score
                            score = (score + rlm_score) / 2
                    except ValueError:
                        pass

            except Exception as e:
                logger.debug(f"[EmailPriority] AragoraRLM analysis failed: {e}")

        return score

    def _score_thread(self, thread_count: int, is_read: bool) -> float:
        """Score based on thread activity."""
        score = 0.5

        # Unread emails get slight boost
        if not is_read:
            score += 0.2

        # Active threads are more important
        if thread_count > 5:
            score = min(1.0, score + 0.3)
        elif thread_count > 2:
            score = min(1.0, score + 0.1)

        return score

    def _generate_reason(
        self,
        factors: Dict[str, float],
        from_address: str,
        subject: str,
    ) -> str:
        """Generate human-readable explanation for the score."""
        reasons = []

        # Find dominant factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)

        for factor_name, factor_score in sorted_factors[:2]:
            if factor_score >= 0.8:
                if factor_name == "sender":
                    reasons.append(f"From important sender ({from_address})")
                elif factor_name == "gmail_signals":
                    reasons.append("Marked as important by Gmail")
                elif factor_name == "urgency":
                    reasons.append("Contains urgent keywords")
                elif factor_name == "content":
                    reasons.append("Matches your interests")
                elif factor_name == "thread":
                    reasons.append("Active conversation")
            elif factor_score <= 0.3:
                if factor_name == "sender":
                    reasons.append("From low-priority sender")
                elif factor_name == "gmail_signals":
                    reasons.append("Categorized as promotional")

        if not reasons:
            total = sum(factors.values()) / len(factors)
            if total >= 0.6:
                reasons.append("Moderately important based on multiple signals")
            else:
                reasons.append("Lower priority - no urgent signals detected")

        return "; ".join(reasons)

    async def score_batch(
        self,
        emails: List[Dict[str, Any]],
    ) -> List[EmailPriorityScore]:
        """
        Score multiple emails in batch.

        Args:
            emails: List of email dicts with id, subject, from_address, snippet, etc.

        Returns:
            List of EmailPriorityScore in same order
        """
        results = []

        for email in emails:
            score = await self.score_email(
                email_id=email.get("id", ""),
                subject=email.get("subject", ""),
                from_address=email.get("from_address", email.get("from", "")),
                snippet=email.get("snippet", ""),
                body_text=email.get("body_text", ""),
                labels=email.get("labels", []),
                is_read=email.get("is_read", False),
                is_starred=email.get("is_starred", False),
                thread_count=email.get("thread_count", 1),
            )
            results.append(score)

        return results


class EmailFeedbackLearner:
    """
    Learn user preferences from email interactions.

    Updates ContinuumMemory based on user actions like:
    - Opening emails
    - Replying to emails
    - Starring/archiving
    - Deleting without reading
    """

    def __init__(self, user_id: str, memory: Optional[Any] = None):
        """
        Initialize feedback learner.

        Args:
            user_id: User ID for memory storage
            memory: ContinuumMemory instance (optional)
        """
        self.user_id = user_id
        self._memory = memory

    async def _get_memory(self) -> Any:
        """Get or create ContinuumMemory instance."""
        if self._memory is None:
            try:
                from aragora.memory.continuum import ContinuumMemory

                self._memory = ContinuumMemory(user_id=self.user_id)  # type: ignore[call-arg]
                await self._memory.initialize()  # type: ignore[attr-defined]
            except ImportError:
                logger.debug("[EmailFeedback] ContinuumMemory not available")
                return None
        return self._memory

    async def record_interaction(
        self,
        email_id: str,
        action: str,
        from_address: str,
        subject: str,
        labels: Optional[List[str]] = None,
    ) -> bool:
        """
        Record a user interaction with an email.

        Args:
            email_id: Email ID
            action: One of: opened, replied, starred, archived, deleted, snoozed
            from_address: Sender email
            subject: Email subject
            labels: Gmail labels

        Returns:
            True if recorded successfully
        """
        memory = await self._get_memory()
        if not memory:
            logger.debug("[EmailFeedback] No memory available for recording")
            return False

        labels = labels or []

        # Build memory content based on action
        if action == "replied":
            content = f"User replied to email from {from_address} about '{subject}'"
            importance = 0.9
            tier = "fast"  # Quick recall for recent contacts

        elif action == "starred":
            content = f"User starred email from {from_address} about '{subject}'"
            importance = 0.8
            tier = "fast"

        elif action == "opened":
            content = f"User opened email from {from_address}"
            importance = 0.5
            tier = "fast"

        elif action == "archived":
            content = f"User archived email from {from_address}"
            importance = 0.3
            tier = "medium"

        elif action == "deleted":
            content = f"User deleted email from {from_address} about '{subject}'"
            importance = 0.2
            tier = "medium"  # Remember to deprioritize similar

        elif action == "snoozed":
            content = f"User snoozed email from {from_address} - wants to revisit"
            importance = 0.6
            tier = "fast"

        else:
            logger.warning(f"[EmailFeedback] Unknown action: {action}")
            return False

        try:
            await memory.store(
                content=content,
                tier=tier,
                importance=importance,
                metadata={
                    "type": "email_interaction",
                    "action": action,
                    "email_id": email_id,
                    "from_address": from_address,
                    "subject": subject,
                    "labels": labels,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.debug(f"[EmailFeedback] Recorded {action} for {email_id}")
            return True

        except Exception as e:
            logger.error(f"[EmailFeedback] Failed to record interaction: {e}")
            return False

    async def consolidate_preferences(self) -> bool:
        """
        Consolidate recent interactions into longer-term preferences.

        Should be called periodically (e.g., daily) to move patterns
        from fast tier to slower tiers.
        """
        memory = await self._get_memory()
        if not memory:
            return False

        try:
            # Trigger memory consolidation
            if hasattr(memory, "consolidate"):
                await memory.consolidate(self.user_id)
                logger.info(f"[EmailFeedback] Consolidated preferences for {self.user_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"[EmailFeedback] Consolidation failed: {e}")
            return False


__all__ = [
    "EmailPriorityAnalyzer",
    "EmailPriorityScore",
    "EmailFeedbackLearner",
    "UserEmailPreferences",
]
