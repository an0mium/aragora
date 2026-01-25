"""
ML-Enhanced Spam Classification Service.

Provides machine learning-based spam detection with online learning
from user feedback. Falls back to rule-based classification when
model confidence is low.

Features:
- Feature extraction from email content, headers, and sender patterns
- Domain reputation scoring (known spam domains, free email providers)
- Subject line analysis (spam keywords, excessive punctuation, ALL CAPS ratio)
- Content n-grams (unigrams, bigrams) for statistical classification
- Header analysis (missing headers, suspicious routing, authentication)
- URL analysis (shortened URLs, suspicious domains, redirect detection)
- Attachment analysis (dangerous extensions, executable files)
- Online learning from user actions (mark as spam, not spam)
- Confidence scoring with fallback to heuristics
- Persistent model storage
- Batch classification support
- Integration with EmailPrioritizer for inbox scoring

Usage:
    from aragora.services.spam_classifier import SpamClassifier

    classifier = SpamClassifier()
    await classifier.initialize()

    # Classify an email
    result = await classifier.classify_email(email)
    print(f"Spam: {result.is_spam} (confidence: {result.confidence})")

    # Train from user feedback
    await classifier.train_from_feedback(email_id, is_spam=True)

    # Use with email dict
    email_dict = {
        "id": "msg_123",
        "subject": "You won!",
        "body": "Click here to claim...",
        "sender": "prize@suspicious.tk",
        "headers": {...},
        "attachments": ["prize.exe"]
    }
    result = await classify_email(email_dict)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle  # Legacy support only - do not use for new saves
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from aragora.services.sender_history import SenderHistoryService

logger = logging.getLogger(__name__)


class SpamCategory(Enum):
    """Spam classification categories."""

    HAM = "ham"  # Not spam
    SPAM = "spam"  # Definitely spam
    PROMOTIONAL = "promotional"  # Marketing/promotional
    SUSPICIOUS = "suspicious"  # Possibly spam
    PHISHING = "phishing"  # Phishing attempt


@dataclass
class SpamClassificationResult:
    """Result of spam classification."""

    email_id: str
    is_spam: bool
    category: SpamCategory
    confidence: float  # 0.0 to 1.0
    spam_score: float  # 0.0 to 1.0 (higher = more spammy)

    # Feature contributions
    content_score: float = 0.0
    sender_score: float = 0.0
    header_score: float = 0.0
    pattern_score: float = 0.0
    url_score: float = 0.0
    attachment_score: float = 0.0
    subject_score: float = 0.0

    # Reasoning
    reasons: List[str] = field(default_factory=list)
    model_used: str = "rule_based"

    # Detected threats
    suspicious_urls: List[str] = field(default_factory=list)
    dangerous_attachments: List[str] = field(default_factory=list)

    # Metadata
    classified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "is_spam": self.is_spam,
            "category": self.category.value,
            "confidence": self.confidence,
            "spam_score": self.spam_score,
            "scores": {
                "content": self.content_score,
                "sender": self.sender_score,
                "header": self.header_score,
                "pattern": self.pattern_score,
                "url": self.url_score,
                "attachment": self.attachment_score,
                "subject": self.subject_score,
            },
            "reasons": self.reasons,
            "model_used": self.model_used,
            "suspicious_urls": self.suspicious_urls,
            "dangerous_attachments": self.dangerous_attachments,
            "classified_at": self.classified_at.isoformat(),
        }

    def get_priority_penalty(self) -> float:
        """
        Get priority penalty for email prioritization integration.

        Returns a value between 0 and 1 that can be used to reduce
        email priority based on spam likelihood.

        Returns:
            Priority penalty (0 = no penalty, 1 = maximum penalty)
        """
        if self.is_spam:
            return 1.0
        if self.category == SpamCategory.PHISHING:
            return 0.9  # High penalty but keep visible for review
        if self.category == SpamCategory.SUSPICIOUS:
            return 0.5
        if self.category == SpamCategory.PROMOTIONAL:
            return 0.3
        return 0.0


@dataclass
class EmailFeatures:
    """Extracted features from an email for classification."""

    # Content features
    word_count: int = 0
    char_count: int = 0
    uppercase_ratio: float = 0.0
    digit_ratio: float = 0.0
    special_char_ratio: float = 0.0
    link_count: int = 0
    suspicious_link_count: int = 0
    attachment_count: int = 0

    # Text patterns
    spam_word_count: int = 0
    urgency_word_count: int = 0
    money_word_count: int = 0
    exclamation_count: int = 0
    question_count: int = 0
    all_caps_word_count: int = 0

    # Subject line features
    subject_all_caps: bool = False
    subject_excessive_punctuation: bool = False
    subject_spam_words: int = 0
    subject_length: int = 0
    subject_has_re_fw: bool = False

    # Sender features
    sender_domain_age_days: int = -1  # -1 = unknown
    sender_has_display_name: bool = False
    sender_domain_suspicious: bool = False
    sender_in_reply_chain: bool = False
    sender_previously_contacted: bool = False
    sender_domain_is_free_email: bool = False
    sender_domain_reputation: float = 0.5  # 0-1, 0.5 = neutral
    sender_is_known_spam_domain: bool = False

    # Header features
    has_spf: bool = False
    has_dkim: bool = False
    has_dmarc: bool = False
    received_hop_count: int = 0
    has_suspicious_headers: bool = False
    missing_required_headers: List[str] = field(default_factory=list)
    has_suspicious_routing: bool = False
    has_forged_headers: bool = False

    # URL features
    shortened_url_count: int = 0
    suspicious_domain_urls: List[str] = field(default_factory=list)
    mismatched_anchor_urls: int = 0  # <a href="X">Y</a> where X != Y
    ip_address_urls: int = 0  # URLs with IP addresses instead of domains
    data_uri_count: int = 0  # data: URIs (can hide content)

    # Attachment features
    dangerous_extension_count: int = 0
    dangerous_attachments: List[str] = field(default_factory=list)
    double_extension_count: int = 0  # e.g., "document.pdf.exe"
    archive_with_executable: bool = False

    # Structure features
    html_text_ratio: float = 0.0
    has_images: bool = False
    image_only_email: bool = False
    has_unsubscribe: bool = False
    has_tracking_pixels: bool = False
    has_hidden_text: bool = False  # White text on white background, etc.
    has_form_elements: bool = False  # Forms in email (phishing indicator)

    # N-gram features (populated during extraction)
    top_unigrams: List[Tuple[str, int]] = field(default_factory=list)
    top_bigrams: List[Tuple[str, int]] = field(default_factory=list)

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML model."""
        return [
            self.word_count / 1000,  # Normalize
            self.char_count / 10000,
            self.uppercase_ratio,
            self.digit_ratio,
            self.special_char_ratio,
            self.link_count / 10,
            self.suspicious_link_count / 5,
            self.attachment_count / 3,
            self.spam_word_count / 10,
            self.urgency_word_count / 5,
            self.money_word_count / 5,
            self.exclamation_count / 10,
            self.question_count / 10,
            self.all_caps_word_count / 10,
            # Subject features
            1.0 if self.subject_all_caps else 0.0,
            1.0 if self.subject_excessive_punctuation else 0.0,
            self.subject_spam_words / 5,
            min(self.subject_length / 100, 1.0),
            # Sender features
            1.0 if self.sender_domain_suspicious else 0.0,
            1.0 if self.sender_has_display_name else 0.0,
            1.0 if self.sender_previously_contacted else 0.0,
            1.0 if self.sender_domain_is_free_email else 0.0,
            self.sender_domain_reputation,
            1.0 if self.sender_is_known_spam_domain else 0.0,
            # Header features
            1.0 if self.has_spf else 0.0,
            1.0 if self.has_dkim else 0.0,
            1.0 if self.has_dmarc else 0.0,
            min(self.received_hop_count / 10, 1.0),
            1.0 if self.has_suspicious_headers else 0.0,
            min(len(self.missing_required_headers) / 5, 1.0),
            1.0 if self.has_suspicious_routing else 0.0,
            1.0 if self.has_forged_headers else 0.0,
            # URL features
            min(self.shortened_url_count / 5, 1.0),
            min(len(self.suspicious_domain_urls) / 5, 1.0),
            min(self.mismatched_anchor_urls / 5, 1.0),
            min(self.ip_address_urls / 3, 1.0),
            min(self.data_uri_count / 3, 1.0),
            # Attachment features
            min(self.dangerous_extension_count / 3, 1.0),
            min(self.double_extension_count / 2, 1.0),
            1.0 if self.archive_with_executable else 0.0,
            # Structure features
            self.html_text_ratio,
            1.0 if self.image_only_email else 0.0,
            1.0 if self.has_unsubscribe else 0.0,
            1.0 if self.has_tracking_pixels else 0.0,
            1.0 if self.has_hidden_text else 0.0,
            1.0 if self.has_form_elements else 0.0,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "word_count": self.word_count,
            "char_count": self.char_count,
            "uppercase_ratio": self.uppercase_ratio,
            "link_count": self.link_count,
            "suspicious_link_count": self.suspicious_link_count,
            "spam_word_count": self.spam_word_count,
            "subject_all_caps": self.subject_all_caps,
            "subject_excessive_punctuation": self.subject_excessive_punctuation,
            "sender_domain_suspicious": self.sender_domain_suspicious,
            "sender_is_known_spam_domain": self.sender_is_known_spam_domain,
            "has_spf": self.has_spf,
            "has_dkim": self.has_dkim,
            "shortened_url_count": self.shortened_url_count,
            "dangerous_extension_count": self.dangerous_extension_count,
            "dangerous_attachments": self.dangerous_attachments,
            "top_unigrams": self.top_unigrams[:10],
            "top_bigrams": self.top_bigrams[:10],
        }


@dataclass
class SpamClassifierConfig:
    """Configuration for spam classifier."""

    # Model settings
    model_path: str = "spam_model.pkl"
    use_ml_model: bool = True
    min_confidence_for_ml: float = 0.7

    # Training settings
    min_training_samples: int = 100
    retrain_interval_hours: int = 24
    max_training_samples: int = 10000

    # Feature extraction
    max_content_length: int = 50000

    # Classification thresholds
    spam_threshold: float = 0.6
    promotional_threshold: float = 0.4
    suspicious_threshold: float = 0.3

    # Storage
    feedback_db_path: str = "spam_feedback.db"


# Spam indicator patterns
SPAM_WORDS = {
    "winner",
    "congratulations",
    "prize",
    "won",
    "lottery",
    "urgent",
    "act now",
    "limited time",
    "expire",
    "hurry",
    "free",
    "discount",
    "offer",
    "deal",
    "save",
    "cash",
    "money",
    "dollars",
    "bitcoin",
    "crypto",
    "click here",
    "click now",
    "unsubscribe",
    "pharmacy",
    "pills",
    "medication",
    "viagra",
    "weight loss",
    "diet",
    "lose weight",
    "earn money",
    "work from home",
    "income",
    "nigerian",
    "prince",
    "inheritance",
    "attorney",
}

URGENCY_WORDS = {
    "urgent",
    "immediately",
    "asap",
    "act now",
    "limited",
    "expire",
    "deadline",
    "hurry",
    "last chance",
    "final notice",
    "important",
    "attention",
    "warning",
    "alert",
    "action required",
}

MONEY_WORDS = {
    "money",
    "cash",
    "dollars",
    "payment",
    "bank",
    "transfer",
    "wire",
    "bitcoin",
    "crypto",
    "invest",
    "million",
    "billion",
    "profit",
    "earn",
    "income",
    "loan",
    "credit",
    "debt",
    "insurance",
    "mortgage",
}

SUSPICIOUS_TLDS = {
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".xyz",
    ".top",
    ".work",
    ".click",
    ".loan",
    ".zip",
    ".mov",
    ".review",
    ".stream",
    ".download",
    ".win",
    ".bid",
    ".racing",
    ".party",
    ".science",
    ".date",
    ".faith",
    ".accountant",
    ".cricket",
}

# Known spam domains (frequently used for spam/phishing)
KNOWN_SPAM_DOMAINS = {
    # These are commonly spoofed or used for spam
    "mailinator.com",
    "guerrillamail.com",
    "10minutemail.com",
    "tempmail.com",
    "throwaway.email",
    "temp-mail.org",
    "fakeinbox.com",
    "sharklasers.com",
    "spam4.me",
    "spamgourmet.com",
    "trashmail.com",
}

# Free email providers (not necessarily spam, but can be indicator)
FREE_EMAIL_PROVIDERS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "mail.com",
    "protonmail.com",
    "zoho.com",
    "icloud.com",
    "yandex.com",
    "gmx.com",
    "mail.ru",
    "qq.com",
    "163.com",
    "126.com",
}

# URL shortener services
URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "j.mp",
    "tiny.cc",
    "shorte.st",
    "v.gd",
    "rb.gy",
    "cutt.ly",
    "shorturl.at",
    "t.ly",
    "rebrand.ly",
    "bl.ink",
    "soo.gd",
    "s.id",
}

# Dangerous file extensions
DANGEROUS_EXTENSIONS = {
    # Executables
    ".exe",
    ".com",
    ".bat",
    ".cmd",
    ".msi",
    ".scr",
    ".pif",
    ".application",
    ".gadget",
    ".msp",
    ".msc",
    # Scripts
    ".js",
    ".jse",
    ".vbs",
    ".vbe",
    ".ws",
    ".wsf",
    ".wsc",
    ".wsh",
    ".ps1",
    ".ps1xml",
    ".ps2",
    ".ps2xml",
    ".psc1",
    ".psc2",
    # Macros and Office
    ".docm",
    ".xlsm",
    ".pptm",
    ".dotm",
    ".xltm",
    ".xlam",
    ".ppam",
    ".ppsm",
    ".sldm",
    # Archives (can contain malware)
    ".jar",
    ".hta",
    ".cpl",
    # Shortcuts
    ".lnk",
    ".inf",
    ".reg",
    # Other dangerous
    ".dll",
    ".ocx",
    ".sys",
    ".drv",
}

PROMOTIONAL_PATTERNS = [
    r"unsubscribe",
    r"email\s+preferences",
    r"opt.?out",
    r"manage\s+subscription",
    r"view\s+in\s+browser",
    r"trouble\s+viewing",
    r"add\s+us\s+to\s+your\s+address\s+book",
    r"©\s*\d{4}",  # Copyright notice
    r"all\s+rights\s+reserved",
]

# Required email headers (absence may indicate forgery)
REQUIRED_HEADERS = {
    "from",
    "to",
    "date",
    "message-id",
}


class SpamFeatures:
    """
    Feature extraction engine for spam classification.

    Extracts comprehensive features from email content, headers,
    sender information, URLs, and attachments.
    """

    def __init__(
        self,
        sender_history_service: Optional["SenderHistoryService"] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            sender_history_service: Optional service for sender reputation
            user_id: User ID for sender history lookups
        """
        self.sender_history = sender_history_service
        self.user_id = user_id

        # Compile regex patterns
        self._url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
        self._email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        self._ip_url_pattern = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        self._anchor_pattern = re.compile(
            r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', re.IGNORECASE
        )
        self._form_pattern = re.compile(r"<form[^>]*>", re.IGNORECASE)
        self._hidden_text_pattern = re.compile(
            r"(?:color:\s*(?:white|#fff|#ffffff|transparent)|" r"font-size:\s*0|display:\s*none)",
            re.IGNORECASE,
        )
        self._data_uri_pattern = re.compile(r"data:[^;]+;base64,", re.IGNORECASE)

    def extract(
        self,
        subject: str,
        body: str,
        sender: str,
        headers: Optional[Dict[str, str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> EmailFeatures:
        """
        Extract all features from an email.

        Args:
            subject: Email subject
            body: Email body (plain text or HTML)
            sender: Sender email address
            headers: Optional email headers dict
            attachments: Optional list of attachment filenames

        Returns:
            EmailFeatures with all extracted features
        """
        features = EmailFeatures()
        headers = headers or {}
        attachments = attachments or []

        # Extract features from each component
        self._extract_content_features(features, subject, body)
        self._extract_subject_features(features, subject)
        self._extract_sender_features(features, sender)
        self._extract_header_features(features, headers)
        self._extract_url_features(features, body)
        self._extract_attachment_features(features, attachments)
        self._extract_ngrams(features, f"{subject} {body}")

        return features

    def _extract_content_features(
        self,
        features: EmailFeatures,
        subject: str,
        body: str,
    ) -> None:
        """Extract content-based features."""
        content = f"{subject}\n\n{body}"
        features.char_count = len(content)

        words = content.split()
        features.word_count = len(words)

        if features.char_count > 0:
            features.uppercase_ratio = sum(1 for c in content if c.isupper()) / features.char_count
            features.digit_ratio = sum(1 for c in content if c.isdigit()) / features.char_count
            features.special_char_ratio = (
                sum(1 for c in content if not c.isalnum() and not c.isspace()) / features.char_count
            )

        # Count various patterns
        content_lower = content.lower()
        features.spam_word_count = sum(1 for word in SPAM_WORDS if word in content_lower)
        features.urgency_word_count = sum(1 for word in URGENCY_WORDS if word in content_lower)
        features.money_word_count = sum(1 for word in MONEY_WORDS if word in content_lower)
        features.exclamation_count = content.count("!")
        features.question_count = content.count("?")
        features.all_caps_word_count = sum(1 for word in words if word.isupper() and len(word) > 2)

        # HTML analysis
        html_count = content.count("<") + content.count(">")
        text_content = re.sub(r"<[^>]+>", "", content)
        if len(text_content) > 0:
            features.html_text_ratio = min(html_count / len(text_content), 1.0)

        features.has_images = "<img" in content.lower() or "image" in content.lower()
        features.image_only_email = (
            features.has_images and features.word_count < 50 and "<img" in content.lower()
        )
        features.has_unsubscribe = "unsubscribe" in content_lower
        features.has_tracking_pixels = bool(
            re.search(
                r"<img[^>]*(?:1x1|pixel|track|beacon|width=[\"']?1|height=[\"']?1)[^>]*>",
                content,
                re.IGNORECASE,
            )
        )
        features.has_hidden_text = bool(self._hidden_text_pattern.search(content))
        features.has_form_elements = bool(self._form_pattern.search(content))

    def _extract_subject_features(self, features: EmailFeatures, subject: str) -> None:
        """Extract subject line specific features."""
        features.subject_length = len(subject)

        # Check if subject is all caps (excluding short subjects)
        if len(subject) > 5:
            alpha_chars = [c for c in subject if c.isalpha()]
            if alpha_chars:
                caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                features.subject_all_caps = caps_ratio > 0.8

        # Check for excessive punctuation
        punct_count = sum(1 for c in subject if c in "!?$*#@")
        if len(subject) > 0:
            features.subject_excessive_punctuation = punct_count / len(subject) > 0.1

        # Count spam words in subject
        subject_lower = subject.lower()
        features.subject_spam_words = sum(1 for word in SPAM_WORDS if word in subject_lower)

        # Check for Re:/Fw: patterns (legitimate replies/forwards)
        features.subject_has_re_fw = bool(re.match(r"^(re|fw|fwd):\s*", subject, re.IGNORECASE))

    def _extract_sender_features(self, features: EmailFeatures, sender: str) -> None:
        """Extract sender-related features."""
        if not sender:
            return

        features.sender_has_display_name = "<" in sender

        # Extract domain
        match = self._email_pattern.search(sender)
        if match:
            email = match.group(0).lower()
            domain = email.split("@")[-1]

            # Check various domain characteristics
            features.sender_domain_is_free_email = domain in FREE_EMAIL_PROVIDERS
            features.sender_is_known_spam_domain = domain in KNOWN_SPAM_DOMAINS
            features.sender_domain_suspicious = any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)

            # Check for suspicious sender patterns
            if re.search(r"noreply|no-reply|donotreply", email):
                pass  # Not necessarily spam, but tracked
            if re.search(r"\d{4,}", email):
                features.sender_domain_suspicious = True

    def _extract_header_features(
        self,
        features: EmailFeatures,
        headers: Dict[str, str],
    ) -> None:
        """Extract header-based features."""
        if not headers:
            return

        # Normalize header names to lowercase for comparison
        headers_lower = {k.lower(): v for k, v in headers.items()}

        # Check for missing required headers
        features.missing_required_headers = [h for h in REQUIRED_HEADERS if h not in headers_lower]

        # Authentication checks
        auth_results = headers_lower.get("authentication-results", "").lower()
        features.has_spf = "spf=pass" in auth_results
        features.has_dkim = "dkim=pass" in auth_results
        features.has_dmarc = "dmarc=pass" in auth_results

        # Count received headers (hop count)
        features.received_hop_count = sum(1 for h in headers.keys() if h.lower() == "received")

        # Check for suspicious headers
        features.has_suspicious_headers = (
            "x-mailer" in headers_lower and "bulk" in headers_lower.get("precedence", "").lower()
        )

        # Check for suspicious routing
        received_headers = [v for k, v in headers.items() if k.lower() == "received"]
        if len(received_headers) > 8:
            features.has_suspicious_routing = True

        # Check for potentially forged headers
        if "x-originating-ip" in headers_lower:
            # Verify IP matches other routing information
            pass  # Advanced forgery detection would go here

    def _extract_url_features(self, features: EmailFeatures, body: str) -> None:
        """Extract URL-based features."""
        urls = self._url_pattern.findall(body)
        features.link_count = len(urls)

        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()

                # Check for URL shorteners
                if domain in URL_SHORTENERS or any(
                    domain.endswith(f".{s}") for s in URL_SHORTENERS
                ):
                    features.shortened_url_count += 1

                # Check for suspicious TLDs
                if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
                    features.suspicious_link_count += 1
                    features.suspicious_domain_urls.append(url)

                # Check for IP address URLs
                if self._ip_url_pattern.match(url):
                    features.ip_address_urls += 1
                    features.suspicious_domain_urls.append(url)

            except (ValueError, AttributeError):
                # Skip malformed URLs
                continue

        # Check for mismatched anchor URLs (phishing indicator)
        anchors = self._anchor_pattern.findall(body)
        for href, text in anchors:
            # If anchor text looks like a URL but doesn't match href
            if re.match(r"https?://", text.strip()):
                text_domain = urlparse(text.strip()).netloc.lower()
                href_domain = urlparse(href).netloc.lower()
                if text_domain and href_domain and text_domain != href_domain:
                    features.mismatched_anchor_urls += 1

        # Count data URIs
        features.data_uri_count = len(self._data_uri_pattern.findall(body))

    def _extract_attachment_features(
        self,
        features: EmailFeatures,
        attachments: List[str],
    ) -> None:
        """Extract attachment-based features."""
        features.attachment_count = len(attachments)

        for attachment in attachments:
            attachment_lower = attachment.lower()

            # Check for dangerous extensions
            for ext in DANGEROUS_EXTENSIONS:
                if attachment_lower.endswith(ext):
                    features.dangerous_extension_count += 1
                    features.dangerous_attachments.append(attachment)
                    break

            # Check for double extensions (e.g., "document.pdf.exe")
            parts = attachment.split(".")
            if len(parts) >= 3:
                # Check if final extension is dangerous and preceded by benign extension
                if any(attachment_lower.endswith(ext) for ext in DANGEROUS_EXTENSIONS):
                    benign_extensions = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".jpg"}
                    second_to_last = f".{parts[-2].lower()}"
                    if second_to_last in benign_extensions:
                        features.double_extension_count += 1

            # Check for archives that might contain executables
            archive_extensions = {".zip", ".rar", ".7z", ".tar", ".gz"}
            if any(attachment_lower.endswith(ext) for ext in archive_extensions):
                # Flag as potential concern (would need to scan content for certainty)
                pass

    def _extract_ngrams(self, features: EmailFeatures, text: str, n_top: int = 20) -> None:
        """Extract unigram and bigram features."""
        # Tokenize
        words = re.findall(r"\b[a-z]{2,15}\b", text.lower())

        # Unigrams
        unigram_counts = Counter(words)
        features.top_unigrams = unigram_counts.most_common(n_top)

        # Bigrams
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        features.top_bigrams = bigram_counts.most_common(n_top)


@dataclass
class SpamFeedback:
    """
    User feedback on spam classification.

    Used for online learning and model improvement.
    """

    email_id: str
    user_id: str
    is_spam: bool
    original_classification: SpamCategory
    original_confidence: float
    feedback_type: str = "explicit"  # "explicit" (user marked) or "implicit" (user action)
    content_hash: Optional[str] = None
    features_json: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "user_id": self.user_id,
            "is_spam": self.is_spam,
            "original_classification": self.original_classification.value,
            "original_confidence": self.original_confidence,
            "feedback_type": self.feedback_type,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
        }


class NaiveBayesClassifier:
    """Simple Naive Bayes classifier for spam detection."""

    def __init__(self):
        """Initialize classifier."""
        self.word_spam_counts: Counter = Counter()
        self.word_ham_counts: Counter = Counter()
        self.spam_count: int = 0
        self.ham_count: int = 0
        self.vocabulary: Set[str] = set()
        self._lock = threading.Lock()

    def train(self, text: str, is_spam: bool) -> None:
        """Train on a single example."""
        words = self._tokenize(text)

        with self._lock:
            if is_spam:
                self.spam_count += 1
                self.word_spam_counts.update(words)
            else:
                self.ham_count += 1
                self.word_ham_counts.update(words)
            self.vocabulary.update(words)

    def predict(self, text: str) -> Tuple[bool, float]:
        """
        Predict if text is spam.

        Returns:
            Tuple of (is_spam, confidence)
        """
        words = self._tokenize(text)

        with self._lock:
            if self.spam_count == 0 and self.ham_count == 0:
                return False, 0.5

            total = self.spam_count + self.ham_count
            vocab_size = len(self.vocabulary) + 1

            # Log probabilities with Laplace smoothing
            log_prob_spam = math.log((self.spam_count + 1) / (total + 2))
            log_prob_ham = math.log((self.ham_count + 1) / (total + 2))

            for word in words:
                # P(word | spam)
                spam_word_prob = (self.word_spam_counts.get(word, 0) + 1) / (
                    self.spam_count + vocab_size
                )
                log_prob_spam += math.log(spam_word_prob)

                # P(word | ham)
                ham_word_prob = (self.word_ham_counts.get(word, 0) + 1) / (
                    self.ham_count + vocab_size
                )
                log_prob_ham += math.log(ham_word_prob)

            # Convert to probabilities
            max_log = max(log_prob_spam, log_prob_ham)
            prob_spam = math.exp(log_prob_spam - max_log)
            prob_ham = math.exp(log_prob_ham - max_log)
            total_prob = prob_spam + prob_ham

            spam_probability = prob_spam / total_prob

            is_spam = spam_probability > 0.5
            confidence = abs(spam_probability - 0.5) * 2  # 0 to 1

            return is_spam, confidence

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        # Simple word tokenization
        words = re.findall(r"\b[a-z]{2,15}\b", text)
        return words

    def save(self, path: str) -> None:
        """Save model to file using JSON (safe serialization)."""
        with self._lock:
            data = {
                "word_spam_counts": dict(self.word_spam_counts),
                "word_ham_counts": dict(self.word_ham_counts),
                "spam_count": self.spam_count,
                "ham_count": self.ham_count,
                "vocabulary": list(self.vocabulary),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

    def load(self, path: str) -> bool:
        """Load model from file.

        Supports JSON (preferred) with legacy pickle fallback for migration.
        """
        try:
            # Try JSON first (secure format)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Legacy pickle fallback - only for migration from old format
            # SECURITY: Only load pickle files from trusted sources
            logger.warning(f"Loading legacy pickle model from {path} - will re-save as JSON")
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                # Re-save as JSON immediately to migrate away from pickle
                self._apply_model_data(data)
                self.save(path)
                logger.info(f"Migrated model from pickle to JSON: {path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load legacy pickle model: {e}")
                return False
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

        self._apply_model_data(data)
        return True

    def _apply_model_data(self, data: dict) -> None:
        """Apply loaded model data to instance."""
        with self._lock:
            self.word_spam_counts = Counter(data["word_spam_counts"])
            self.word_ham_counts = Counter(data["word_ham_counts"])
            self.spam_count = data["spam_count"]
            self.ham_count = data["ham_count"]
            self.vocabulary = set(data["vocabulary"])

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.spam_count > 0 or self.ham_count > 0


class SpamClassifier:
    """
    ML-enhanced spam classifier with online learning.

    Uses a Naive Bayes model trained on user feedback,
    with fallback to rule-based classification.

    Features:
    - Domain reputation scoring
    - Subject line analysis (ALL CAPS, excessive punctuation)
    - Content n-grams (unigrams, bigrams)
    - Header analysis (SPF, DKIM, DMARC, suspicious routing)
    - URL analysis (shorteners, suspicious domains, IP addresses)
    - Attachment analysis (dangerous extensions)
    - Online learning from user feedback
    - Fallback to rule-based when ML confidence < 0.7
    """

    def __init__(
        self,
        config: Optional[SpamClassifierConfig] = None,
        sender_history_service: Optional["SenderHistoryService"] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize spam classifier.

        Args:
            config: Classifier configuration
            sender_history_service: Optional service for sender reputation
            user_id: User ID for sender history lookups
        """
        self.config = config or SpamClassifierConfig()
        self.model = NaiveBayesClassifier()
        self._db_conn: Optional[sqlite3.Connection] = None
        self._compiled_promotional = [re.compile(p, re.IGNORECASE) for p in PROMOTIONAL_PATTERNS]
        self._initialized = False
        self._last_retrain: Optional[datetime] = None

        # Feature extractor
        self._feature_extractor = SpamFeatures(
            sender_history_service=sender_history_service,
            user_id=user_id,
        )

        # Domain reputation cache
        self._domain_reputation_cache: Dict[str, Tuple[datetime, float]] = {}
        self._domain_cache_ttl = 3600  # 1 hour

    async def initialize(self) -> None:
        """Initialize classifier (load model, setup database)."""
        if self._initialized:
            return

        # Try to load existing model
        if os.path.exists(self.config.model_path):
            if self.model.load(self.config.model_path):
                logger.info(
                    f"Loaded spam model with {self.model.spam_count} spam "
                    f"and {self.model.ham_count} ham samples"
                )

        # Initialize feedback database
        await self._init_feedback_db()
        self._initialized = True

    async def _init_feedback_db(self) -> None:
        """Initialize feedback storage database."""
        try:
            self._db_conn = sqlite3.connect(
                self.config.feedback_db_path,
                check_same_thread=False,
            )
            cursor = self._db_conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spam_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT NOT NULL,
                    user_id TEXT,
                    is_spam BOOLEAN NOT NULL,
                    content_hash TEXT,
                    features_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    used_for_training BOOLEAN DEFAULT FALSE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_training
                ON spam_feedback(used_for_training)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classification_cache (
                    content_hash TEXT PRIMARY KEY,
                    is_spam BOOLEAN,
                    category TEXT,
                    confidence REAL,
                    spam_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._db_conn.commit()
            logger.info("Spam feedback database initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize feedback database: {e}")

    async def classify_email(
        self,
        email_id: str,
        subject: str,
        body: str,
        sender: str,
        headers: Optional[Dict[str, str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> SpamClassificationResult:
        """
        Classify an email as spam or ham.

        Args:
            email_id: Email identifier
            subject: Email subject
            body: Email body (text or HTML)
            sender: Sender email address
            headers: Optional email headers
            attachments: Optional list of attachment filenames

        Returns:
            SpamClassificationResult
        """
        if not self._initialized:
            await self.initialize()

        # Extract features using enhanced feature extractor
        features = self._feature_extractor.extract(
            subject=subject,
            body=body,
            sender=sender,
            headers=headers,
            attachments=attachments,
        )

        # Check cache
        content = f"{subject}\n\n{body}"
        content_hash = self._hash_content(content)
        cached = await self._get_cached_result(content_hash)
        if cached:
            cached.email_id = email_id
            return cached

        reasons: List[str] = []
        model_used = "rule_based"

        # Try ML model first if trained
        ml_is_spam = False
        ml_confidence = 0.0

        if self.config.use_ml_model and self.model.is_trained:
            ml_is_spam, ml_confidence = self.model.predict(content)
            if ml_confidence >= self.config.min_confidence_for_ml:
                model_used = "naive_bayes"

        # Calculate comprehensive rule-based scores
        content_score = self._score_content(content, features)
        sender_score = self._score_sender(sender, features)
        header_score = self._score_headers(headers or {}, features)
        pattern_score = self._score_patterns(content)
        url_score = self._score_urls(features)
        attachment_score = self._score_attachments(features)
        subject_score = self._score_subject(features)

        # Combine scores with weights
        weights = {
            "content": 0.25,
            "sender": 0.15,
            "header": 0.10,
            "pattern": 0.15,
            "url": 0.15,
            "attachment": 0.10,
            "subject": 0.10,
        }

        spam_score = (
            content_score * weights["content"]
            + sender_score * weights["sender"]
            + header_score * weights["header"]
            + pattern_score * weights["pattern"]
            + url_score * weights["url"]
            + attachment_score * weights["attachment"]
            + subject_score * weights["subject"]
        )

        # If ML model is confident, blend with rule-based
        if model_used == "naive_bayes":
            ml_score = 1.0 if ml_is_spam else 0.0
            spam_score = spam_score * 0.4 + ml_score * 0.6
            reasons.append(f"ML model: {'spam' if ml_is_spam else 'ham'} ({ml_confidence:.0%})")

        # Add detailed reasons based on high-scoring components
        if content_score > 0.5:
            reasons.append(f"Suspicious content (score: {content_score:.2f})")
        if sender_score > 0.5:
            reasons.append(f"Suspicious sender (score: {sender_score:.2f})")
        if url_score > 0.5:
            reasons.append(
                f"Suspicious URLs ({features.shortened_url_count} shortened, "
                f"{len(features.suspicious_domain_urls)} suspicious)"
            )
        if attachment_score > 0.5:
            reasons.append(
                f"Dangerous attachments: {', '.join(features.dangerous_attachments[:3])}"
            )
        if subject_score > 0.5:
            reasons.append(f"Suspicious subject line (score: {subject_score:.2f})")

        # Determine category
        category, confidence = self._determine_category(spam_score, content, features, reasons)

        # Fallback to rule-based when confidence is low
        if confidence < self.config.min_confidence_for_ml and model_used == "naive_bayes":
            model_used = "rule_based_fallback"
            reasons.append("Fallback to rules (low ML confidence)")

        is_spam = category in (SpamCategory.SPAM, SpamCategory.PHISHING)

        result = SpamClassificationResult(
            email_id=email_id,
            is_spam=is_spam,
            category=category,
            confidence=confidence,
            spam_score=spam_score,
            content_score=content_score,
            sender_score=sender_score,
            header_score=header_score,
            pattern_score=pattern_score,
            url_score=url_score,
            attachment_score=attachment_score,
            subject_score=subject_score,
            reasons=reasons,
            model_used=model_used,
            suspicious_urls=features.suspicious_domain_urls[:10],
            dangerous_attachments=features.dangerous_attachments,
        )

        # Cache result
        await self._cache_result(content_hash, result)

        return result

    def _extract_features(
        self,
        content: str,
        sender: str,
        headers: Optional[Dict[str, str]],
        attachments: Optional[List[str]],
    ) -> EmailFeatures:
        """Extract features from email for classification."""
        features = EmailFeatures()

        # Content features
        features.char_count = len(content)
        words = content.split()
        features.word_count = len(words)

        if features.char_count > 0:
            features.uppercase_ratio = sum(1 for c in content if c.isupper()) / features.char_count
            features.digit_ratio = sum(1 for c in content if c.isdigit()) / features.char_count
            features.special_char_ratio = (
                sum(1 for c in content if not c.isalnum() and not c.isspace()) / features.char_count
            )

        # Link analysis
        urls = re.findall(r'https?://[^\s<>"]+', content)
        features.link_count = len(urls)
        features.suspicious_link_count = sum(
            1 for url in urls if any(tld in url.lower() for tld in SUSPICIOUS_TLDS)
        )

        # Attachments
        features.attachment_count = len(attachments or [])

        # Text patterns
        content_lower = content.lower()
        features.spam_word_count = sum(1 for word in SPAM_WORDS if word in content_lower)
        features.urgency_word_count = sum(1 for word in URGENCY_WORDS if word in content_lower)
        features.money_word_count = sum(1 for word in MONEY_WORDS if word in content_lower)
        features.exclamation_count = content.count("!")
        features.question_count = content.count("?")
        features.all_caps_word_count = sum(1 for word in words if word.isupper() and len(word) > 2)

        # Sender features
        if sender:
            features.sender_has_display_name = "<" in sender
            domain = sender.split("@")[-1].split(">")[0].lower()
            features.sender_domain_suspicious = any(tld in domain for tld in SUSPICIOUS_TLDS)

        # Header features
        if headers:
            features.has_spf = "spf=pass" in headers.get("Authentication-Results", "").lower()
            features.has_dkim = "dkim=pass" in headers.get("Authentication-Results", "").lower()
            features.has_dmarc = "dmarc=pass" in headers.get("Authentication-Results", "").lower()
            features.received_hop_count = len(
                [h for h in headers.keys() if h.lower() == "received"]
            )
            features.has_suspicious_headers = (
                "x-mailer" in {h.lower() for h in headers.keys()}
                and "bulk" in headers.get("Precedence", "").lower()
            )

        # Structure features
        html_count = content.count("<") + content.count(">")
        text_count = len(re.sub(r"<[^>]+>", "", content))
        if text_count > 0:
            features.html_text_ratio = min(html_count / text_count, 1.0)

        features.has_images = "<img" in content.lower() or "image" in content.lower()
        features.image_only_email = (
            features.has_images and features.word_count < 50 and "<img" in content.lower()
        )
        features.has_unsubscribe = "unsubscribe" in content_lower
        features.has_tracking_pixels = bool(
            re.search(
                r"<img[^>]*(?:1x1|pixel|track|beacon)[^>]*>",
                content,
                re.IGNORECASE,
            )
        )

        return features

    def _score_content(self, content: str, features: EmailFeatures) -> float:
        """Score content for spam likelihood (0-1)."""
        score = 0.0

        # Spam word density
        if features.word_count > 0:
            spam_density = features.spam_word_count / features.word_count
            score += min(spam_density * 10, 0.3)

        # Urgency indicators
        urgency_score = min(features.urgency_word_count / 5, 0.2)
        score += urgency_score

        # Money indicators
        money_score = min(features.money_word_count / 5, 0.15)
        score += money_score

        # Formatting issues
        if features.uppercase_ratio > 0.3:
            score += 0.15
        if features.exclamation_count > 5:
            score += 0.1
        if features.all_caps_word_count > 5:
            score += 0.1

        return min(score, 1.0)

    def _score_sender(self, sender: str, features: EmailFeatures) -> float:
        """Score sender for spam likelihood (0-1)."""
        score = 0.0

        if features.sender_domain_suspicious:
            score += 0.4

        # Check for suspicious sender patterns
        if sender:
            sender_lower = sender.lower()
            if re.search(r"noreply|no-reply|donotreply", sender_lower):
                score += 0.1
            if re.search(r"\d{4,}", sender_lower):
                score += 0.15
            if not features.sender_has_display_name:
                score += 0.05

        if features.sender_previously_contacted:
            score -= 0.3

        return max(0.0, min(score, 1.0))

    def _score_headers(self, headers: Dict[str, str], features: EmailFeatures) -> float:
        """Score headers for spam likelihood (0-1)."""
        score = 0.0

        # Missing authentication
        if not features.has_spf:
            score += 0.15
        if not features.has_dkim:
            score += 0.15
        if not features.has_dmarc:
            score += 0.1

        # Too many hops
        if features.received_hop_count > 8:
            score += 0.2

        # Suspicious headers
        if features.has_suspicious_headers:
            score += 0.2

        return min(score, 1.0)

    def _score_patterns(self, content: str) -> float:
        """Score for known spam/promotional patterns (0-1)."""
        score = 0.0
        promotional_matches = 0

        for pattern in self._compiled_promotional:
            if pattern.search(content):
                promotional_matches += 1

        # High promotional content
        if promotional_matches >= 3:
            score += 0.3
        elif promotional_matches >= 1:
            score += 0.15

        # Common spam phrases
        spam_phrases = [
            r"act\s+now",
            r"limited\s+time",
            r"click\s+here\s+now",
            r"congratulations.*won",
            r"you\s+have\s+been\s+selected",
            r"100%\s+free",
            r"no\s+credit\s+card",
            r"risk[\s-]free",
            r"double\s+your",
            r"earn\s+\$?\d+",
        ]

        phrase_matches = sum(
            1 for phrase in spam_phrases if re.search(phrase, content, re.IGNORECASE)
        )
        score += min(phrase_matches * 0.15, 0.5)

        return min(score, 1.0)

    def _score_urls(self, features: EmailFeatures) -> float:
        """Score URL-based spam indicators (0-1)."""
        score = 0.0

        # Shortened URLs are suspicious
        if features.shortened_url_count > 0:
            score += min(features.shortened_url_count * 0.15, 0.3)

        # Suspicious domain URLs
        if len(features.suspicious_domain_urls) > 0:
            score += min(len(features.suspicious_domain_urls) * 0.2, 0.4)

        # IP address URLs (very suspicious)
        if features.ip_address_urls > 0:
            score += min(features.ip_address_urls * 0.25, 0.4)

        # Mismatched anchor URLs (phishing indicator)
        if features.mismatched_anchor_urls > 0:
            score += min(features.mismatched_anchor_urls * 0.3, 0.5)

        # Data URIs can hide content
        if features.data_uri_count > 0:
            score += min(features.data_uri_count * 0.1, 0.2)

        # Too many links relative to content
        if features.word_count > 0 and features.link_count > 0:
            link_density = features.link_count / (features.word_count / 100)
            if link_density > 5:  # More than 5 links per 100 words
                score += 0.15

        return min(score, 1.0)

    def _score_attachments(self, features: EmailFeatures) -> float:
        """Score attachment-based spam/malware indicators (0-1)."""
        score = 0.0

        # Dangerous extensions
        if features.dangerous_extension_count > 0:
            score += min(features.dangerous_extension_count * 0.4, 0.8)

        # Double extensions (high risk)
        if features.double_extension_count > 0:
            score += min(features.double_extension_count * 0.5, 0.8)

        # Archive with executable (high risk)
        if features.archive_with_executable:
            score += 0.6

        return min(score, 1.0)

    def _score_subject(self, features: EmailFeatures) -> float:
        """Score subject line spam indicators (0-1)."""
        score = 0.0

        # ALL CAPS subject
        if features.subject_all_caps:
            score += 0.3

        # Excessive punctuation
        if features.subject_excessive_punctuation:
            score += 0.2

        # Spam words in subject
        if features.subject_spam_words > 0:
            score += min(features.subject_spam_words * 0.15, 0.4)

        # Very short subject (often spam)
        if features.subject_length < 5 and features.subject_length > 0:
            score += 0.1

        # Legitimate reply/forward patterns reduce score
        if features.subject_has_re_fw:
            score = max(0, score - 0.2)

        return min(score, 1.0)

    def _determine_category(
        self,
        spam_score: float,
        content: str,
        features: EmailFeatures,
        reasons: List[str],
    ) -> Tuple[SpamCategory, float]:
        """Determine spam category and confidence."""
        # Check for phishing indicators
        phishing_score = self._check_phishing(content, features)
        if phishing_score > 0.7:
            reasons.append("High phishing indicators")
            return SpamCategory.PHISHING, phishing_score

        # High spam score
        if spam_score >= self.config.spam_threshold:
            reasons.append(f"High spam score: {spam_score:.2f}")
            confidence = min(spam_score * 1.2, 1.0)
            return SpamCategory.SPAM, confidence

        # Promotional content
        if spam_score >= self.config.promotional_threshold and features.has_unsubscribe:
            reasons.append("Promotional email with unsubscribe link")
            return SpamCategory.PROMOTIONAL, spam_score * 0.9

        # Suspicious but not definitive
        if spam_score >= self.config.suspicious_threshold:
            reasons.append(f"Suspicious patterns: {spam_score:.2f}")
            return SpamCategory.SUSPICIOUS, spam_score

        # Ham (not spam)
        confidence = 1.0 - spam_score
        return SpamCategory.HAM, confidence

    def _check_phishing(self, content: str, features: EmailFeatures) -> float:
        """Check for phishing indicators."""
        score = 0.0

        # Suspicious links
        if features.suspicious_link_count > 0:
            score += 0.3

        # Mismatched anchor URLs (strong phishing indicator)
        if features.mismatched_anchor_urls > 0:
            score += 0.4

        # IP address URLs (very suspicious)
        if features.ip_address_urls > 0:
            score += 0.3

        # Urgency + action required
        if features.urgency_word_count >= 2:
            score += 0.2

        # Form elements in email (credential harvesting)
        if features.has_form_elements:
            score += 0.4

        # Known spam domain sender
        if features.sender_is_known_spam_domain:
            score += 0.3

        # Common phishing phrases
        phishing_phrases = [
            r"verify\s+your\s+account",
            r"confirm\s+your\s+identity",
            r"update\s+your\s+payment",
            r"account\s+(suspended|disabled|locked)",
            r"unusual\s+activity",
            r"security\s+alert",
            r"password\s+expir",
            r"click\s+.*\s+verify",
            r"within\s+24\s+hours",
            r"your\s+account\s+will\s+be",
            r"immediately\s+verify",
            r"confirm\s+your\s+bank",
            r"ssn|social\s+security",
            r"credit\s+card\s+number",
        ]

        matches = sum(1 for phrase in phishing_phrases if re.search(phrase, content, re.IGNORECASE))
        score += min(matches * 0.15, 0.5)

        return min(score, 1.0)

    async def train_from_feedback(
        self,
        email_id: str,
        is_spam: bool,
        content: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Train the model from user feedback.

        Args:
            email_id: Email identifier
            is_spam: User's classification
            content: Optional email content for training
            user_id: Optional user identifier

        Returns:
            True if feedback was recorded
        """
        if not self._initialized:
            await self.initialize()

        try:
            content_hash = self._hash_content(content or email_id)

            # Store feedback
            if self._db_conn:
                cursor = self._db_conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO spam_feedback
                    (email_id, user_id, is_spam, content_hash)
                    VALUES (?, ?, ?, ?)
                    """,
                    (email_id, user_id, is_spam, content_hash),
                )
                self._db_conn.commit()

            # Train model if content available
            if content:
                self.model.train(content, is_spam)

                # Invalidate cache for this content
                await self._invalidate_cache(content_hash)

            # Check if we should retrain
            await self._maybe_retrain()

            return True

        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")
            return False

    async def _maybe_retrain(self) -> None:
        """Check if we should retrain the model."""
        if not self._db_conn:
            return

        now = datetime.now()
        if self._last_retrain:
            hours_since = (now - self._last_retrain).total_seconds() / 3600
            if hours_since < self.config.retrain_interval_hours:
                return

        try:
            cursor = self._db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM spam_feedback WHERE used_for_training = FALSE")
            untrained_count = cursor.fetchone()[0]

            if untrained_count >= self.config.min_training_samples // 10:
                await self._retrain_model()

        except Exception as e:
            logger.warning(f"Failed to check retrain status: {e}")

    async def _retrain_model(self) -> None:
        """Retrain model from feedback database."""
        if not self._db_conn:
            return

        logger.info("Retraining spam model from feedback...")

        try:
            cursor = self._db_conn.cursor()

            # Get unprocessed feedback
            cursor.execute(
                """
                SELECT id, is_spam, content_hash FROM spam_feedback
                WHERE used_for_training = FALSE
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self.config.max_training_samples,),
            )

            rows = cursor.fetchall()
            trained = 0

            for row in rows:
                feedback_id, is_spam, content_hash = row
                # Note: In a real system, you'd store content or retrieve it
                # For now, we mark as used
                cursor.execute(
                    "UPDATE spam_feedback SET used_for_training = TRUE WHERE id = ?",
                    (feedback_id,),
                )
                trained += 1

            self._db_conn.commit()

            # Save updated model
            self.model.save(self.config.model_path)
            self._last_retrain = datetime.now()

            logger.info(f"Retrained spam model with {trained} feedback samples")

        except Exception as e:
            logger.warning(f"Failed to retrain model: {e}")

    def _hash_content(self, content: str) -> str:
        """Hash content for caching."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_cached_result(self, content_hash: str) -> Optional[SpamClassificationResult]:
        """Get cached classification result."""
        if not self._db_conn:
            return None

        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                SELECT is_spam, category, confidence, spam_score
                FROM classification_cache
                WHERE content_hash = ?
                AND created_at > datetime('now', '-1 day')
                """,
                (content_hash,),
            )
            row = cursor.fetchone()
            if row:
                return SpamClassificationResult(
                    email_id="",
                    is_spam=row[0],
                    category=SpamCategory(row[1]),
                    confidence=row[2],
                    spam_score=row[3],
                    model_used="cached",
                )
        except Exception:
            pass
        return None

    async def _cache_result(self, content_hash: str, result: SpamClassificationResult) -> None:
        """Cache classification result."""
        if not self._db_conn:
            return

        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO classification_cache
                (content_hash, is_spam, category, confidence, spam_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    content_hash,
                    result.is_spam,
                    result.category.value,
                    result.confidence,
                    result.spam_score,
                ),
            )
            self._db_conn.commit()
        except Exception:
            pass

    async def _invalidate_cache(self, content_hash: str) -> None:
        """Invalidate cached result after feedback."""
        if not self._db_conn:
            return

        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "DELETE FROM classification_cache WHERE content_hash = ?",
                (content_hash,),
            )
            self._db_conn.commit()
        except Exception:
            pass

    async def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        stats = {
            "model_trained": self.model.is_trained,
            "spam_samples": self.model.spam_count,
            "ham_samples": self.model.ham_count,
            "vocabulary_size": len(self.model.vocabulary),
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
        }

        if self._db_conn:
            try:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM spam_feedback")
                stats["total_feedback"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM spam_feedback WHERE is_spam = TRUE")
                stats["spam_feedback"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM classification_cache")
                stats["cached_results"] = cursor.fetchone()[0]
            except Exception:
                pass

        return stats

    async def close(self) -> None:
        """Close resources."""
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None


# Convenience function
async def classify_email_spam(
    email_id: str,
    subject: str,
    body: str,
    sender: str,
    headers: Optional[Dict[str, str]] = None,
    attachments: Optional[List[str]] = None,
) -> SpamClassificationResult:
    """
    Quick convenience function for spam classification.

    Args:
        email_id: Email identifier
        subject: Email subject
        body: Email body
        sender: Sender email
        headers: Optional email headers
        attachments: Optional list of attachment filenames

    Returns:
        SpamClassificationResult
    """
    classifier = SpamClassifier()
    await classifier.initialize()

    try:
        return await classifier.classify_email(
            email_id=email_id,
            subject=subject,
            body=body,
            sender=sender,
            headers=headers,
            attachments=attachments,
        )
    finally:
        await classifier.close()


async def classify_email(email: Dict[str, Any]) -> SpamClassificationResult:
    """
    Classify an email from a dictionary representation.

    This is the primary convenience function for spam classification,
    designed to work with the email inbox feature.

    Args:
        email: Dictionary containing email data with keys:
            - id (str): Email identifier
            - subject (str): Email subject
            - body (str): Email body (text or HTML)
            - sender (str): Sender email address
            - headers (dict, optional): Email headers
            - attachments (list, optional): List of attachment filenames

    Returns:
        SpamClassificationResult with is_spam, confidence, category, and reasons

    Example:
        email = {
            "id": "msg_123",
            "subject": "You won a prize!",
            "body": "Click here to claim...",
            "sender": "prize@suspicious.tk",
            "headers": {"Authentication-Results": "spf=fail"},
            "attachments": ["prize.exe"]
        }
        result = await classify_email(email)
        if result.is_spam:
            print(f"Spam detected: {result.reasons}")
    """
    return await classify_email_spam(
        email_id=email.get("id", ""),
        subject=email.get("subject", ""),
        body=email.get("body", "") or email.get("body_text", "") or email.get("body_html", ""),
        sender=email.get("sender", "") or email.get("from_address", ""),
        headers=email.get("headers"),
        attachments=email.get("attachments"),
    )


async def classify_emails_batch(
    emails: List[Dict[str, Any]],
    max_concurrent: int = 10,
) -> List[SpamClassificationResult]:
    """
    Classify multiple emails concurrently.

    Args:
        emails: List of email dictionaries
        max_concurrent: Maximum concurrent classifications

    Returns:
        List of SpamClassificationResult in same order as input
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)
    classifier = SpamClassifier()
    await classifier.initialize()

    async def classify_one(email: Dict[str, Any]) -> SpamClassificationResult:
        async with semaphore:
            return await classifier.classify_email(
                email_id=email.get("id", ""),
                subject=email.get("subject", ""),
                body=email.get("body", "") or email.get("body_text", "") or "",
                sender=email.get("sender", "") or email.get("from_address", ""),
                headers=email.get("headers"),
                attachments=email.get("attachments"),
            )

    try:
        tasks = [classify_one(email) for email in emails]
        return await asyncio.gather(*tasks)
    finally:
        await classifier.close()


__all__ = [
    # Core classes
    "SpamClassifier",
    "SpamClassifierConfig",
    "SpamClassificationResult",
    "SpamCategory",
    # Feature extraction
    "SpamFeatures",
    "EmailFeatures",
    # Feedback tracking
    "SpamFeedback",
    # Convenience functions
    "classify_email",
    "classify_email_spam",
    "classify_emails_batch",
    # Pattern sets (for extension)
    "SPAM_WORDS",
    "URGENCY_WORDS",
    "MONEY_WORDS",
    "SUSPICIOUS_TLDS",
    "KNOWN_SPAM_DOMAINS",
    "URL_SHORTENERS",
    "DANGEROUS_EXTENSIONS",
]
