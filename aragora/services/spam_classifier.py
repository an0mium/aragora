"""
ML-Enhanced Spam Classification Service.

Provides machine learning-based spam detection with online learning
from user feedback. Falls back to rule-based classification when
model confidence is low.

Features:
- Feature extraction from email content, headers, and sender patterns
- Online learning from user actions (mark as spam, not spam)
- Confidence scoring with fallback to heuristics
- Persistent model storage
- Batch classification support

Usage:
    from aragora.services.spam_classifier import SpamClassifier

    classifier = SpamClassifier()
    await classifier.initialize()

    # Classify an email
    result = await classifier.classify_email(email)
    print(f"Spam: {result.is_spam} (confidence: {result.confidence})")

    # Train from user feedback
    await classifier.train_from_feedback(email_id, is_spam=True)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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

    # Reasoning
    reasons: List[str] = field(default_factory=list)
    model_used: str = "rule_based"

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
            },
            "reasons": self.reasons,
            "model_used": self.model_used,
            "classified_at": self.classified_at.isoformat(),
        }


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

    # Sender features
    sender_domain_age_days: int = -1  # -1 = unknown
    sender_has_display_name: bool = False
    sender_domain_suspicious: bool = False
    sender_in_reply_chain: bool = False
    sender_previously_contacted: bool = False

    # Header features
    has_spf: bool = False
    has_dkim: bool = False
    has_dmarc: bool = False
    received_hop_count: int = 0
    has_suspicious_headers: bool = False

    # Structure features
    html_text_ratio: float = 0.0
    has_images: bool = False
    image_only_email: bool = False
    has_unsubscribe: bool = False
    has_tracking_pixels: bool = False

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
            1.0 if self.sender_domain_suspicious else 0.0,
            1.0 if self.sender_has_display_name else 0.0,
            1.0 if self.sender_previously_contacted else 0.0,
            1.0 if self.has_spf else 0.0,
            1.0 if self.has_dkim else 0.0,
            1.0 if self.has_dmarc else 0.0,
            min(self.received_hop_count / 10, 1.0),
            1.0 if self.has_suspicious_headers else 0.0,
            self.html_text_ratio,
            1.0 if self.image_only_email else 0.0,
            1.0 if self.has_unsubscribe else 0.0,
            1.0 if self.has_tracking_pixels else 0.0,
        ]


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
        """Save model to file."""
        with self._lock:
            data = {
                "word_spam_counts": dict(self.word_spam_counts),
                "word_ham_counts": dict(self.word_ham_counts),
                "spam_count": self.spam_count,
                "ham_count": self.ham_count,
                "vocabulary": list(self.vocabulary),
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def load(self, path: str) -> bool:
        """Load model from file."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self.word_spam_counts = Counter(data["word_spam_counts"])
                self.word_ham_counts = Counter(data["word_ham_counts"])
                self.spam_count = data["spam_count"]
                self.ham_count = data["ham_count"]
                self.vocabulary = set(data["vocabulary"])
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self.spam_count > 0 or self.ham_count > 0


class SpamClassifier:
    """
    ML-enhanced spam classifier with online learning.

    Uses a Naive Bayes model trained on user feedback,
    with fallback to rule-based classification.
    """

    def __init__(self, config: Optional[SpamClassifierConfig] = None):
        """
        Initialize spam classifier.

        Args:
            config: Classifier configuration
        """
        self.config = config or SpamClassifierConfig()
        self.model = NaiveBayesClassifier()
        self._db_conn: Optional[sqlite3.Connection] = None
        self._compiled_promotional = [re.compile(p, re.IGNORECASE) for p in PROMOTIONAL_PATTERNS]
        self._initialized = False
        self._last_retrain: Optional[datetime] = None

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

        # Extract features
        content = f"{subject}\n\n{body}"
        features = self._extract_features(content, sender, headers, attachments)

        # Check cache
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

        # Calculate rule-based scores
        content_score = self._score_content(content, features)
        sender_score = self._score_sender(sender, features)
        header_score = self._score_headers(headers or {}, features)
        pattern_score = self._score_patterns(content)

        # Combine scores
        weights = {
            "content": 0.35,
            "sender": 0.25,
            "header": 0.15,
            "pattern": 0.25,
        }

        spam_score = (
            content_score * weights["content"]
            + sender_score * weights["sender"]
            + header_score * weights["header"]
            + pattern_score * weights["pattern"]
        )

        # If ML model is confident, blend with rule-based
        if model_used == "naive_bayes":
            ml_score = 1.0 if ml_is_spam else 0.0
            spam_score = spam_score * 0.4 + ml_score * 0.6
            reasons.append(f"ML model: {'spam' if ml_is_spam else 'ham'} ({ml_confidence:.0%})")

        # Determine category
        category, confidence = self._determine_category(spam_score, content, features, reasons)

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
            reasons=reasons,
            model_used=model_used,
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

        # Urgency + action required
        if features.urgency_word_count >= 2:
            score += 0.2

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
        ]

        matches = sum(1 for phrase in phishing_phrases if re.search(phrase, content, re.IGNORECASE))
        score += min(matches * 0.2, 0.5)

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
) -> SpamClassificationResult:
    """
    Quick convenience function for spam classification.

    Args:
        email_id: Email identifier
        subject: Email subject
        body: Email body
        sender: Sender email

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
        )
    finally:
        await classifier.close()


__all__ = [
    "SpamClassifier",
    "SpamClassifierConfig",
    "SpamClassificationResult",
    "SpamCategory",
    "EmailFeatures",
    "classify_email_spam",
]
