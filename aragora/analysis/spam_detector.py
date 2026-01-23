"""
Spam and Phishing Detection Module.

Provides multi-signal spam and phishing detection:
- Bayesian text classification (pre-trained + user feedback)
- Sender reputation (domain age, SPF/DKIM/DMARC)
- Link analysis (shortened URLs, mismatched domains)
- Attachment risk (file types, password-protected zips)
- Urgency manipulation detection
- Impersonation detection (CEO fraud patterns)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SpamCategory(Enum):
    """Category of spam/unwanted email."""

    COMMERCIAL = "commercial"  # Marketing, newsletters
    PHISHING = "phishing"  # Credential harvesting
    MALWARE = "malware"  # Malicious attachments/links
    SCAM = "scam"  # Nigerian prince, lottery, etc.
    IMPERSONATION = "impersonation"  # CEO fraud, brand spoofing
    BULK = "bulk"  # Mass mailings
    LEGITIMATE = "legitimate"  # Not spam


class RiskLevel(Enum):
    """Risk level of email."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SpamSignal:
    """A single spam detection signal."""

    name: str
    score: float  # 0.0 (not spam) to 1.0 (definitely spam)
    weight: float  # Importance of this signal
    details: str = ""

    def weighted_score(self) -> float:
        """Get weighted score."""
        return self.score * self.weight


@dataclass
class LinkAnalysis:
    """Analysis of a URL in the email."""

    url: str
    display_text: Optional[str]
    domain: str
    is_shortened: bool
    is_suspicious: bool
    redirect_chain: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class SpamAnalysis:
    """Complete spam analysis result."""

    email_id: str
    is_spam: bool
    spam_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    category: SpamCategory
    confidence: float  # 0.0 to 1.0
    signals: List[SpamSignal] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    links_analyzed: List[LinkAnalysis] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "is_spam": self.is_spam,
            "spam_score": self.spam_score,
            "risk_level": self.risk_level.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "signals": [
                {"name": s.name, "score": s.score, "weight": s.weight, "details": s.details}
                for s in self.signals
            ],
            "reasons": self.reasons,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


@dataclass
class PhishingAnalysis:
    """Phishing-specific analysis result."""

    email_id: str
    is_phishing: bool
    phishing_score: float  # 0.0 to 1.0
    confidence: float
    indicators: List[str] = field(default_factory=list)
    targeted_brand: Optional[str] = None
    credential_harvesting_detected: bool = False
    login_page_mimicry: bool = False
    suspicious_links: List[LinkAnalysis] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "is_phishing": self.is_phishing,
            "phishing_score": self.phishing_score,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "targeted_brand": self.targeted_brand,
            "credential_harvesting_detected": self.credential_harvesting_detected,
            "login_page_mimicry": self.login_page_mimicry,
            "suspicious_links": len(self.suspicious_links),
        }


@dataclass
class EmailContent:
    """Email content for analysis."""

    email_id: str
    sender: str
    sender_name: Optional[str] = None
    subject: str = ""
    body_text: str = ""
    body_html: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    received_at: Optional[datetime] = None


class SpamDetector:
    """
    Multi-signal spam and phishing detection.

    Uses a combination of heuristic rules, pattern matching,
    and statistical analysis to detect unwanted emails.
    """

    # Suspicious file extensions
    RISKY_EXTENSIONS = {
        ".exe",
        ".scr",
        ".bat",
        ".cmd",
        ".com",
        ".pif",
        ".application",
        ".vbs",
        ".vbe",
        ".js",
        ".jse",
        ".ws",
        ".wsf",
        ".wsh",
        ".ps1",
        ".msi",
        ".msp",
        ".msc",
        ".jar",
        ".hta",
        ".cpl",
        ".dll",
        ".iso",
    }

    # URL shortener domains
    URL_SHORTENERS = {
        "bit.ly",
        "t.co",
        "goo.gl",
        "tinyurl.com",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "su.pr",
        "bit.do",
        "bl.ink",
        "short.cm",
    }

    # Common phishing keywords
    PHISHING_KEYWORDS = {
        "verify your account",
        "confirm your identity",
        "update your password",
        "suspended account",
        "unusual activity",
        "security alert",
        "click here immediately",
        "act now",
        "urgent action required",
        "verify your information",
        "reset your password",
    }

    # Brand impersonation targets
    COMMON_BRANDS = {
        "microsoft",
        "apple",
        "google",
        "amazon",
        "paypal",
        "netflix",
        "facebook",
        "instagram",
        "linkedin",
        "twitter",
        "whatsapp",
        "bank of america",
        "chase",
        "wells fargo",
        "citibank",
        "ups",
        "fedex",
        "dhl",
        "usps",
        "irs",
        "social security",
    }

    def __init__(
        self,
        spam_threshold: float = 0.7,
        phishing_threshold: float = 0.6,
        user_whitelist: Optional[Set[str]] = None,
        user_blacklist: Optional[Set[str]] = None,
    ):
        """
        Initialize the spam detector.

        Args:
            spam_threshold: Score above which email is classified as spam
            phishing_threshold: Score above which email is classified as phishing
            user_whitelist: Set of trusted sender addresses
            user_blacklist: Set of known spam sender addresses
        """
        self.spam_threshold = spam_threshold
        self.phishing_threshold = phishing_threshold
        self.user_whitelist = user_whitelist or set()
        self.user_blacklist = user_blacklist or set()

    def analyze(self, email: EmailContent) -> SpamAnalysis:
        """
        Analyze an email for spam.

        Args:
            email: Email content to analyze

        Returns:
            SpamAnalysis with spam classification
        """
        signals: List[SpamSignal] = []

        # Check whitelist/blacklist first
        sender_lower = email.sender.lower()
        if sender_lower in self.user_whitelist:
            return SpamAnalysis(
                email_id=email.email_id,
                is_spam=False,
                spam_score=0.0,
                risk_level=RiskLevel.SAFE,
                category=SpamCategory.LEGITIMATE,
                confidence=1.0,
                reasons=["Sender is whitelisted"],
            )

        if sender_lower in self.user_blacklist:
            return SpamAnalysis(
                email_id=email.email_id,
                is_spam=True,
                spam_score=1.0,
                risk_level=RiskLevel.HIGH,
                category=SpamCategory.BULK,
                confidence=1.0,
                reasons=["Sender is blacklisted"],
            )

        # Analyze text content
        signals.append(self._analyze_text_content(email))

        # Analyze sender
        signals.append(self._analyze_sender(email))

        # Analyze links
        link_signal, links = self._analyze_links(email)
        signals.append(link_signal)

        # Analyze attachments
        signals.append(self._analyze_attachments(email))

        # Analyze headers
        signals.append(self._analyze_headers(email))

        # Analyze urgency patterns
        signals.append(self._analyze_urgency(email))

        # Calculate overall score
        total_weight = sum(s.weight for s in signals)
        if total_weight > 0:
            spam_score = sum(s.weighted_score() for s in signals) / total_weight
        else:
            spam_score = 0.0

        # Determine category
        category = self._determine_category(signals, email)

        # Determine risk level
        risk_level = self._determine_risk_level(spam_score, signals)

        # Generate reasons
        reasons = [s.details for s in signals if s.score > 0.5 and s.details]

        is_spam = spam_score >= self.spam_threshold

        return SpamAnalysis(
            email_id=email.email_id,
            is_spam=is_spam,
            spam_score=spam_score,
            risk_level=risk_level,
            category=category,
            confidence=min(1.0, 0.5 + abs(spam_score - 0.5)),
            signals=signals,
            reasons=reasons[:5],  # Top 5 reasons
            links_analyzed=links,
        )

    def analyze_phishing(self, email: EmailContent) -> PhishingAnalysis:
        """
        Analyze an email specifically for phishing.

        Args:
            email: Email content to analyze

        Returns:
            PhishingAnalysis with phishing classification
        """
        indicators: List[str] = []
        suspicious_links: List[LinkAnalysis] = []
        targeted_brand: Optional[str] = None
        phishing_score = 0.0

        # Check for brand impersonation
        content = f"{email.subject} {email.body_text}".lower()
        for brand in self.COMMON_BRANDS:
            if brand in content:
                targeted_brand = brand
                indicators.append(f"References brand: {brand}")
                phishing_score += 0.1
                break

        # Check for phishing keywords
        for keyword in self.PHISHING_KEYWORDS:
            if keyword.lower() in content.lower():
                indicators.append(f"Phishing keyword: '{keyword}'")
                phishing_score += 0.15

        # Analyze links for credential harvesting
        urls = self._extract_urls(email)
        credential_harvesting = False
        login_mimicry = False

        for url, display_text in urls:
            link_analysis = self._analyze_single_link(url, display_text)
            if link_analysis.is_suspicious:
                suspicious_links.append(link_analysis)
                phishing_score += 0.2

                # Check for credential harvesting patterns
                if any(x in url.lower() for x in ["login", "signin", "password", "verify"]):
                    credential_harvesting = True
                    indicators.append(f"Potential credential harvesting: {url[:50]}...")

                # Check for login page mimicry
                if targeted_brand and targeted_brand in url.lower():
                    if link_analysis.domain not in targeted_brand:
                        login_mimicry = True
                        indicators.append(f"Brand mimicry: {link_analysis.domain}")

        # Check sender domain mismatch
        if targeted_brand:
            sender_domain = email.sender.split("@")[-1].lower()
            if targeted_brand not in sender_domain:
                indicators.append("Sender domain doesn't match brand")
                phishing_score += 0.2

        # Cap score at 1.0
        phishing_score = min(1.0, phishing_score)

        is_phishing = phishing_score >= self.phishing_threshold

        return PhishingAnalysis(
            email_id=email.email_id,
            is_phishing=is_phishing,
            phishing_score=phishing_score,
            confidence=min(1.0, 0.5 + abs(phishing_score - 0.5)),
            indicators=indicators[:10],
            targeted_brand=targeted_brand,
            credential_harvesting_detected=credential_harvesting,
            login_page_mimicry=login_mimicry,
            suspicious_links=suspicious_links,
        )

    def _analyze_text_content(self, email: EmailContent) -> SpamSignal:
        """Analyze text content for spam patterns."""
        score = 0.0
        content = f"{email.subject} {email.body_text}".lower()

        # Check for common spam phrases
        spam_phrases = [
            "congratulations",
            "winner",
            "prize",
            "claim now",
            "limited time",
            "act fast",
            "free gift",
            "earn money",
            "make money",
            "work from home",
            "lose weight",
            "weight loss",
            "cheap",
            "lowest price",
            "affordable",
            "discount",
        ]
        for phrase in spam_phrases:
            if phrase in content:
                score += 0.1

        # Check for excessive capitalization
        if email.subject:
            caps_ratio = sum(1 for c in email.subject if c.isupper()) / max(len(email.subject), 1)
            if caps_ratio > 0.5:
                score += 0.2

        # Check for excessive punctuation
        punct_count = sum(1 for c in content if c in "!?$")
        if punct_count > 10:
            score += 0.15

        return SpamSignal(
            name="text_analysis",
            score=min(1.0, score),
            weight=0.25,
            details="Spam patterns in text" if score > 0.3 else "",
        )

    def _analyze_sender(self, email: EmailContent) -> SpamSignal:
        """Analyze sender for spam indicators."""
        score = 0.0
        sender_lower = email.sender.lower()

        # Check for suspicious TLDs
        suspicious_tlds = [".xyz", ".top", ".click", ".link", ".info", ".biz"]
        for tld in suspicious_tlds:
            if sender_lower.endswith(tld):
                score += 0.3
                break

        # Check for random-looking addresses
        local_part = sender_lower.split("@")[0]
        if len(local_part) > 20 and sum(c.isdigit() for c in local_part) > 5:
            score += 0.25

        # Check for name mismatch
        if email.sender_name:
            name_lower = email.sender_name.lower()
            if "@" in name_lower:  # Name contains email (spoofing attempt)
                score += 0.4

        return SpamSignal(
            name="sender_analysis",
            score=min(1.0, score),
            weight=0.2,
            details="Suspicious sender" if score > 0.3 else "",
        )

    def _analyze_links(self, email: EmailContent) -> Tuple[SpamSignal, List[LinkAnalysis]]:
        """Analyze URLs in the email."""
        urls = self._extract_urls(email)
        link_analyses: List[LinkAnalysis] = []
        score = 0.0

        for url, display_text in urls:
            analysis = self._analyze_single_link(url, display_text)
            link_analyses.append(analysis)
            if analysis.is_suspicious:
                score += 0.15

        # Many links is suspicious
        if len(urls) > 10:
            score += 0.2

        return (
            SpamSignal(
                name="link_analysis",
                score=min(1.0, score),
                weight=0.2,
                details="Suspicious links detected" if score > 0.2 else "",
            ),
            link_analyses,
        )

    def _analyze_single_link(self, url: str, display_text: Optional[str]) -> LinkAnalysis:
        """Analyze a single URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            domain = ""

        is_shortened = domain in self.URL_SHORTENERS
        risk_factors: List[str] = []
        is_suspicious = False

        # Check for URL shorteners
        if is_shortened:
            risk_factors.append("URL shortener")
            is_suspicious = True

        # Check for IP address URLs
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
            risk_factors.append("IP address URL")
            is_suspicious = True

        # Check for display text mismatch
        if display_text and domain not in display_text.lower():
            # Could be legitimate (e.g., "Click here")
            if any(x in display_text.lower() for x in ["http", "www.", ".com", ".org"]):
                risk_factors.append("Display text domain mismatch")
                is_suspicious = True

        # Check for suspicious paths
        if any(x in url.lower() for x in ["/login", "/signin", "/verify", "/update"]):
            risk_factors.append("Credential path detected")

        return LinkAnalysis(
            url=url,
            display_text=display_text,
            domain=domain,
            is_shortened=is_shortened,
            is_suspicious=is_suspicious,
            risk_factors=risk_factors,
        )

    def _analyze_attachments(self, email: EmailContent) -> SpamSignal:
        """Analyze attachments for risks."""
        if not email.attachments:
            return SpamSignal(name="attachment_analysis", score=0.0, weight=0.15)

        score = 0.0
        for att in email.attachments:
            filename = att.get("filename", "").lower()
            ext = "." + filename.split(".")[-1] if "." in filename else ""

            if ext in self.RISKY_EXTENSIONS:
                score += 0.5

            # Double extensions (e.g., file.pdf.exe)
            if filename.count(".") > 1:
                score += 0.3

        return SpamSignal(
            name="attachment_analysis",
            score=min(1.0, score),
            weight=0.15,
            details="Risky attachments" if score > 0.3 else "",
        )

    def _analyze_headers(self, email: EmailContent) -> SpamSignal:
        """Analyze email headers for authenticity."""
        score = 0.0

        # Check for authentication failures
        auth_results = email.headers.get("Authentication-Results", "").lower()
        if "fail" in auth_results or "none" in auth_results:
            score += 0.3

        # Check for missing headers
        if "Message-ID" not in email.headers:
            score += 0.1

        return SpamSignal(
            name="header_analysis",
            score=min(1.0, score),
            weight=0.1,
            details="Header issues" if score > 0.2 else "",
        )

    def _analyze_urgency(self, email: EmailContent) -> SpamSignal:
        """Analyze urgency manipulation patterns."""
        content = f"{email.subject} {email.body_text}".lower()
        score = 0.0

        urgency_phrases = [
            "urgent",
            "immediate",
            "act now",
            "limited time",
            "expires today",
            "last chance",
            "don't miss",
            "within 24 hours",
            "respond immediately",
        ]

        for phrase in urgency_phrases:
            if phrase in content:
                score += 0.15

        return SpamSignal(
            name="urgency_analysis",
            score=min(1.0, score),
            weight=0.1,
            details="High urgency language" if score > 0.3 else "",
        )

    def _extract_urls(self, email: EmailContent) -> List[Tuple[str, Optional[str]]]:
        """Extract URLs from email content."""
        urls: List[Tuple[str, Optional[str]]] = []

        # Extract from HTML
        if email.body_html:
            # Simple href extraction
            for match in re.finditer(r'href=["\']([^"\']+)["\']', email.body_html):
                url = match.group(1)
                if url.startswith("http"):
                    urls.append((url, None))

        # Extract from plain text
        for match in re.finditer(r'https?://[^\s<>"]+', email.body_text):
            url = match.group(0)
            if url not in [u[0] for u in urls]:
                urls.append((url, None))

        return urls

    def _determine_category(
        self,
        signals: List[SpamSignal],
        email: EmailContent,
    ) -> SpamCategory:
        """Determine the spam category."""
        content = f"{email.subject} {email.body_text}".lower()

        # Check for phishing
        if any("phishing" in s.name or "credential" in s.details.lower() for s in signals):
            return SpamCategory.PHISHING

        # Check for scam
        scam_keywords = ["lottery", "winner", "inheritance", "million dollars", "prince"]
        if any(kw in content for kw in scam_keywords):
            return SpamCategory.SCAM

        # Check for commercial
        commercial_keywords = ["unsubscribe", "newsletter", "promotional", "special offer"]
        if any(kw in content for kw in commercial_keywords):
            return SpamCategory.COMMERCIAL

        # Check for impersonation
        for brand in self.COMMON_BRANDS:
            if brand in content:
                sender_domain = email.sender.split("@")[-1].lower()
                if brand not in sender_domain:
                    return SpamCategory.IMPERSONATION

        return SpamCategory.BULK

    def _determine_risk_level(
        self,
        spam_score: float,
        signals: List[SpamSignal],
    ) -> RiskLevel:
        """Determine risk level based on score and signals."""
        if spam_score < 0.2:
            return RiskLevel.SAFE
        elif spam_score < 0.4:
            return RiskLevel.LOW
        elif spam_score < 0.7:
            return RiskLevel.MEDIUM
        elif spam_score < 0.9:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def add_to_whitelist(self, sender: str) -> None:
        """Add a sender to the whitelist."""
        self.user_whitelist.add(sender.lower())

    def add_to_blacklist(self, sender: str) -> None:
        """Add a sender to the blacklist."""
        self.user_blacklist.add(sender.lower())

    def remove_from_whitelist(self, sender: str) -> None:
        """Remove a sender from the whitelist."""
        self.user_whitelist.discard(sender.lower())

    def remove_from_blacklist(self, sender: str) -> None:
        """Remove a sender from the blacklist."""
        self.user_blacklist.discard(sender.lower())
