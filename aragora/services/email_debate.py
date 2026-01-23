"""
Email Debate Service.

Multi-agent deliberation on email prioritization, categorization, and triage.
Uses Arena for structured debate between agents to determine:
- Priority level (urgent, high, normal, low)
- Category assignment
- Action required (reply needed, FYI, etc.)
- Spam/phishing detection

Integrates with:
- PIIRedactor for sanitizing emails before analysis
- SenderHistoryManager for reputation context
- Deliberation templates for structured workflows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailPriority(str, Enum):
    """Priority levels for emails."""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class EmailCategory(str, Enum):
    """Categories for email classification."""

    ACTION_REQUIRED = "action_required"
    REPLY_NEEDED = "reply_needed"
    FYI = "fyi"
    MEETING = "meeting"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    SOCIAL = "social"
    SPAM = "spam"
    PHISHING = "phishing"
    UNKNOWN = "unknown"


@dataclass
class EmailInput:
    """Email data for deliberation."""

    subject: str
    body: str
    sender: str
    received_at: datetime
    recipients: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None
    message_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Convert to context string for debate."""
        parts = [
            f"From: {self.sender}",
            f"Subject: {self.subject}",
            f"Date: {self.received_at.isoformat()}",
        ]
        if self.recipients:
            parts.append(f"To: {', '.join(self.recipients)}")
        if self.cc:
            parts.append(f"CC: {', '.join(self.cc)}")
        if self.attachments:
            parts.append(f"Attachments: {', '.join(self.attachments)}")
        parts.append("")
        parts.append(self.body[:2000])  # Truncate long bodies
        return "\n".join(parts)


@dataclass
class EmailDebateResult:
    """Result of multi-agent email deliberation."""

    message_id: str
    priority: EmailPriority
    category: EmailCategory
    confidence: float
    reasoning: str
    action_items: List[str] = field(default_factory=list)
    suggested_labels: List[str] = field(default_factory=list)
    is_spam: bool = False
    is_phishing: bool = False
    sender_reputation: Optional[float] = None
    debate_id: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "priority": self.priority.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "action_items": self.action_items,
            "suggested_labels": self.suggested_labels,
            "is_spam": self.is_spam,
            "is_phishing": self.is_phishing,
            "sender_reputation": self.sender_reputation,
            "debate_id": self.debate_id,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class BatchEmailResult:
    """Result of batch email deliberation."""

    results: List[EmailDebateResult]
    total_emails: int
    processed_emails: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)

    @property
    def by_priority(self) -> Dict[str, List[EmailDebateResult]]:
        """Group results by priority."""
        grouped: Dict[str, List[EmailDebateResult]] = {}
        for r in self.results:
            key = r.priority.value
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        return grouped

    @property
    def urgent_count(self) -> int:
        """Count of urgent emails."""
        return len([r for r in self.results if r.priority == EmailPriority.URGENT])

    @property
    def action_required_count(self) -> int:
        """Count of emails requiring action."""
        return len([r for r in self.results if r.category == EmailCategory.ACTION_REQUIRED])


class EmailDebateService:
    """
    Service for multi-agent email deliberation.

    Uses Arena to run structured debates on email content
    with multiple AI agents for prioritization and categorization.
    """

    def __init__(
        self,
        agents: Optional[List[str]] = None,
        enable_pii_redaction: bool = True,
        enable_sender_reputation: bool = True,
        preserve_domains: Optional[List[str]] = None,
        fast_mode: bool = True,
    ):
        """
        Initialize email debate service.

        Args:
            agents: Agent identifiers for debate (default: anthropic-api, openai-api)
            enable_pii_redaction: Whether to redact PII before analysis
            enable_sender_reputation: Whether to fetch sender reputation
            preserve_domains: Email domains to NOT redact (company domains)
            fast_mode: Use fast 2-round debates (True) or thorough 5-round (False)
        """
        self.agents = agents or ["anthropic-api", "openai-api"]
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_sender_reputation = enable_sender_reputation
        self.preserve_domains = preserve_domains or []
        self.fast_mode = fast_mode

        self._pii_redactor = None
        self._sender_history = None
        self._debate_factory = None

    def _get_pii_redactor(self):
        """Lazy-load PII redactor."""
        if self._pii_redactor is None and self.enable_pii_redaction:
            try:
                from aragora.services.pii_redactor import PIIRedactor

                self._pii_redactor = PIIRedactor(preserve_domains=self.preserve_domains)
            except ImportError:
                logger.warning("PIIRedactor not available")
        return self._pii_redactor

    def _get_sender_history(self):
        """Lazy-load sender history manager."""
        if self._sender_history is None and self.enable_sender_reputation:
            try:
                from aragora.services.sender_history import SenderHistoryManager

                self._sender_history = SenderHistoryManager()
            except ImportError:
                logger.warning("SenderHistoryManager not available")
        return self._sender_history

    def _get_debate_factory(self):
        """Lazy-load debate factory."""
        if self._debate_factory is None:
            try:
                from aragora.server.debate_factory import DebateFactory

                self._debate_factory = DebateFactory()
            except ImportError:
                logger.warning("DebateFactory not available")
        return self._debate_factory

    def _sanitize_email(self, email: EmailInput) -> EmailInput:
        """Sanitize email content by redacting PII."""
        redactor = self._get_pii_redactor()
        if not redactor:
            return email

        # Redact subject and body
        subject_result = redactor.redact(email.subject)
        body_result = redactor.redact(email.body)

        # Create new email with redacted content
        return EmailInput(
            subject=subject_result.redacted_text,
            body=body_result.redacted_text,
            sender=email.sender,  # Keep sender for reputation lookup
            received_at=email.received_at,
            recipients=email.recipients,
            cc=email.cc,
            thread_id=email.thread_id,
            message_id=email.message_id,
            attachments=email.attachments,
            headers=email.headers,
        )

    async def _get_sender_reputation(self, user_id: str, sender: str) -> Optional[float]:
        """Get sender reputation score."""
        history = self._get_sender_history()
        if not history:
            return None

        try:
            reputation = await history.get_sender_reputation(user_id, sender)
            if reputation:
                return reputation.reputation_score
        except Exception as e:
            logger.debug(f"Failed to get sender reputation: {e}")
        return None

    def _build_debate_prompt(
        self,
        email: EmailInput,
        sender_reputation: Optional[float] = None,
    ) -> str:
        """Build the debate prompt for email analysis."""
        context = email.to_context_string()

        prompt = """Analyze this email and determine:
1. Priority level (urgent, high, normal, low, spam)
2. Category (action_required, reply_needed, fyi, meeting, newsletter, promotional, social, spam, phishing)
3. Whether this requires immediate action
4. Any spam/phishing indicators

Consider:
- Subject urgency indicators (URGENT, ASAP, deadline)
- Sender importance and relationship
- Content relevance to the recipient
- Time sensitivity
- Potential security threats

"""
        if sender_reputation is not None:
            prompt += f"Sender reputation score: {sender_reputation:.2f} (0=unknown, 1=trusted)\n\n"

        prompt += f"EMAIL:\n{context}"

        return prompt

    async def prioritize_email(
        self,
        email: EmailInput,
        user_id: str = "default",
    ) -> EmailDebateResult:
        """
        Run multi-agent vetted decisionmaking to prioritize a single email.

        Args:
            email: Email to analyze
            user_id: User ID for reputation lookup

        Returns:
            EmailDebateResult with priority and category
        """
        start_time = datetime.now(timezone.utc)

        # Sanitize email
        sanitized = self._sanitize_email(email)

        # Get sender reputation
        sender_reputation = await self._get_sender_reputation(user_id, email.sender)

        # Build debate prompt
        prompt = self._build_debate_prompt(sanitized, sender_reputation)

        # Get debate factory
        factory = self._get_debate_factory()
        if not factory:
            # Fallback without full debate
            return self._fallback_prioritization(email, sender_reputation)

        # Create and run debate
        try:
            from aragora.server.debate_factory import DebateConfig

            config = DebateConfig(
                question=prompt,
                agents_str=",".join(self.agents),
                rounds=2 if self.fast_mode else 5,
                consensus="majority",
                debate_format="light" if self.fast_mode else "full",
            )

            arena = factory.create_arena(config)
            result = await arena.run()

            # Parse result
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            return self._parse_debate_result(
                email=email,
                debate_result=result,
                sender_reputation=sender_reputation,
                duration=elapsed,
            )

        except Exception as e:
            logger.error(f"Debate failed: {e}")
            return self._fallback_prioritization(email, sender_reputation)

    def _parse_debate_result(
        self,
        email: EmailInput,
        debate_result: Any,
        sender_reputation: Optional[float],
        duration: float,
    ) -> EmailDebateResult:
        """Parse Arena debate result into EmailDebateResult."""
        # Extract answer from debate result
        answer = getattr(debate_result, "final_answer", "")
        if not answer:
            answer = getattr(debate_result, "answer", "")

        # Detect priority from answer
        priority = EmailPriority.NORMAL
        answer_lower = answer.lower()
        if "urgent" in answer_lower:
            priority = EmailPriority.URGENT
        elif "high" in answer_lower:
            priority = EmailPriority.HIGH
        elif "low" in answer_lower:
            priority = EmailPriority.LOW
        elif "spam" in answer_lower:
            priority = EmailPriority.SPAM

        # Detect category from answer
        category = EmailCategory.UNKNOWN
        if "action" in answer_lower or "required" in answer_lower:
            category = EmailCategory.ACTION_REQUIRED
        elif "reply" in answer_lower:
            category = EmailCategory.REPLY_NEEDED
        elif "fyi" in answer_lower or "information" in answer_lower:
            category = EmailCategory.FYI
        elif "meeting" in answer_lower or "calendar" in answer_lower:
            category = EmailCategory.MEETING
        elif "newsletter" in answer_lower:
            category = EmailCategory.NEWSLETTER
        elif "promotional" in answer_lower or "marketing" in answer_lower:
            category = EmailCategory.PROMOTIONAL
        elif "phishing" in answer_lower:
            category = EmailCategory.PHISHING
        elif "spam" in answer_lower:
            category = EmailCategory.SPAM

        # Get confidence
        confidence = getattr(debate_result, "confidence", 0.7)
        debate_id = getattr(debate_result, "debate_id", None)

        return EmailDebateResult(
            message_id=email.message_id or "",
            priority=priority,
            category=category,
            confidence=confidence,
            reasoning=answer[:500],  # Truncate long reasoning
            is_spam=category in (EmailCategory.SPAM, EmailCategory.PHISHING),
            is_phishing=category == EmailCategory.PHISHING,
            sender_reputation=sender_reputation,
            debate_id=debate_id,
            duration_seconds=duration,
        )

    def _fallback_prioritization(
        self,
        email: EmailInput,
        sender_reputation: Optional[float],
    ) -> EmailDebateResult:
        """Fallback prioritization using simple heuristics."""
        subject_lower = email.subject.lower()

        # Simple heuristics
        priority = EmailPriority.NORMAL
        category = EmailCategory.UNKNOWN

        if any(word in subject_lower for word in ["urgent", "asap", "immediately", "critical"]):
            priority = EmailPriority.URGENT
            category = EmailCategory.ACTION_REQUIRED
        elif any(word in subject_lower for word in ["important", "deadline", "reminder"]):
            priority = EmailPriority.HIGH
        elif any(word in subject_lower for word in ["newsletter", "unsubscribe", "weekly digest"]):
            priority = EmailPriority.LOW
            category = EmailCategory.NEWSLETTER
        elif any(word in subject_lower for word in ["meeting", "invite", "calendar"]):
            category = EmailCategory.MEETING

        # Check for spam indicators
        spam_indicators = ["winner", "lottery", "free gift", "click here"]
        if any(ind in subject_lower for ind in spam_indicators):
            priority = EmailPriority.SPAM
            category = EmailCategory.SPAM

        return EmailDebateResult(
            message_id=email.message_id or "",
            priority=priority,
            category=category,
            confidence=0.5,  # Lower confidence for fallback
            reasoning="Analyzed using heuristic rules (debate unavailable)",
            is_spam=priority == EmailPriority.SPAM,
            sender_reputation=sender_reputation,
            duration_seconds=0.0,
        )

    async def prioritize_batch(
        self,
        emails: List[EmailInput],
        user_id: str = "default",
        max_concurrent: int = 5,
    ) -> BatchEmailResult:
        """
        Run multi-agent vetted decisionmaking on a batch of emails.

        Args:
            emails: List of emails to analyze
            user_id: User ID for reputation lookup
            max_concurrent: Maximum concurrent debates

        Returns:
            BatchEmailResult with all results
        """
        import asyncio

        start_time = datetime.now(timezone.utc)
        results: List[EmailDebateResult] = []
        errors: List[str] = []

        # Process in batches to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_email(email: EmailInput) -> Optional[EmailDebateResult]:
            async with semaphore:
                try:
                    return await self.prioritize_email(email, user_id)
                except Exception as e:
                    errors.append(f"{email.message_id}: {e}")
                    return None

        # Run all tasks
        tasks = [process_email(email) for email in emails]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, EmailDebateResult):
                results.append(result)
            elif isinstance(result, Exception):
                errors.append(str(result))

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return BatchEmailResult(
            results=results,
            total_emails=len(emails),
            processed_emails=len(results),
            duration_seconds=elapsed,
            errors=errors,
        )


# Convenience function
async def prioritize_emails(
    emails: List[Dict[str, Any]],
    user_id: str = "default",
    fast_mode: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to prioritize emails.

    Args:
        emails: List of email dicts with subject, body, sender, received_at
        user_id: User ID for reputation lookup
        fast_mode: Use fast 2-round debates

    Returns:
        List of result dicts with priority and category
    """
    service = EmailDebateService(fast_mode=fast_mode)

    email_inputs = []
    for e in emails:
        email_inputs.append(
            EmailInput(
                subject=e.get("subject", ""),
                body=e.get("body", ""),
                sender=e.get("sender", ""),
                received_at=e.get("received_at", datetime.now(timezone.utc)),
                message_id=e.get("message_id"),
            )
        )

    result = await service.prioritize_batch(email_inputs, user_id)
    return [r.to_dict() for r in result.results]
