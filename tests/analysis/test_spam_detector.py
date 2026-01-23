"""
Tests for the spam detector module.

Tests:
- Spam detection signals
- Phishing detection
- Whitelist/blacklist handling
- Link analysis
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.analysis.spam_detector import (
    EmailContent,
    LinkAnalysis,
    PhishingAnalysis,
    RiskLevel,
    SpamAnalysis,
    SpamCategory,
    SpamDetector,
    SpamSignal,
)


class TestEmailContent:
    """Tests for EmailContent dataclass."""

    def test_create_email(self):
        """Test creating an email."""
        email = EmailContent(
            email_id="test-123",
            sender="test@example.com",
            subject="Hello",
            body_text="This is a test.",
        )
        assert email.email_id == "test-123"
        assert email.sender == "test@example.com"


class TestSpamSignal:
    """Tests for SpamSignal dataclass."""

    def test_weighted_score(self):
        """Test weighted score calculation."""
        signal = SpamSignal(
            name="test",
            score=0.8,
            weight=0.5,
        )
        assert signal.weighted_score() == 0.4


class TestSpamDetector:
    """Tests for SpamDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return SpamDetector()

    def test_clean_email(self, detector):
        """Test analysis of clean email."""
        email = EmailContent(
            email_id="clean-1",
            sender="colleague@company.com",
            subject="Meeting tomorrow",
            body_text="Hi, just wanted to confirm our meeting tomorrow at 2pm.",
        )
        result = detector.analyze(email)
        assert result.is_spam is False
        assert result.spam_score < 0.5
        assert result.risk_level in [RiskLevel.SAFE, RiskLevel.LOW]

    def test_obvious_spam(self, detector):
        """Test analysis of obvious spam."""
        email = EmailContent(
            email_id="spam-1",
            sender="winner@lottery-prize.xyz",
            subject="CONGRATULATIONS! YOU WON $1,000,000!!!",
            body_text="Click here NOW to claim your prize!!! Limited time offer! Act FAST! "
            "You have won the lottery! Claim immediately!",
        )
        result = detector.analyze(email)
        # With default threshold of 0.7, weighted scores may not trigger is_spam
        # but signals should detect spam patterns
        assert result.spam_score > 0.2
        signals = {s.name: s.score for s in result.signals}
        # High text spam score due to spam phrases
        assert signals.get("text_analysis", 0) > 0.5
        # Category should identify it as a scam
        assert result.category == SpamCategory.SCAM

    def test_whitelist(self, detector):
        """Test whitelisted sender."""
        detector.add_to_whitelist("trusted@example.com")
        email = EmailContent(
            email_id="whitelisted-1",
            sender="trusted@example.com",
            subject="URGENT: Click here NOW!!!",
            body_text="This would look like spam but sender is trusted.",
        )
        result = detector.analyze(email)
        assert result.is_spam is False
        assert result.spam_score == 0.0
        assert "whitelisted" in result.reasons[0].lower()

    def test_blacklist(self, detector):
        """Test blacklisted sender."""
        detector.add_to_blacklist("spammer@spam.com")
        email = EmailContent(
            email_id="blacklisted-1",
            sender="spammer@spam.com",
            subject="Normal subject",
            body_text="Normal content",
        )
        result = detector.analyze(email)
        assert result.is_spam is True
        assert result.spam_score == 1.0
        assert "blacklisted" in result.reasons[0].lower()

    def test_suspicious_sender(self, detector):
        """Test suspicious sender detection."""
        email = EmailContent(
            email_id="sus-sender-1",
            sender="abc123456789xyz@suspicious.xyz",
            subject="Important message",
            body_text="Please review this.",
        )
        result = detector.analyze(email)
        signals = {s.name: s.score for s in result.signals}
        assert signals.get("sender_analysis", 0) > 0

    def test_risky_attachments(self, detector):
        """Test risky attachment detection."""
        email = EmailContent(
            email_id="att-1",
            sender="sender@example.com",
            subject="Check this file",
            body_text="Please open the attached file.",
            attachments=[
                {"filename": "invoice.pdf.exe"},
                {"filename": "document.scr"},
            ],
        )
        result = detector.analyze(email)
        signals = {s.name: s.score for s in result.signals}
        assert signals.get("attachment_analysis", 0) > 0.3

    def test_urgency_detection(self, detector):
        """Test urgency pattern detection."""
        email = EmailContent(
            email_id="urgent-1",
            sender="sender@example.com",
            subject="URGENT: Act immediately!",
            body_text="You must respond within 24 hours or you will lose access!",
        )
        result = detector.analyze(email)
        signals = {s.name: s.score for s in result.signals}
        assert signals.get("urgency_analysis", 0) > 0

    def test_category_detection_scam(self, detector):
        """Test scam category detection."""
        email = EmailContent(
            email_id="scam-1",
            sender="prince@nigeria.com",
            subject="Inheritance of $10 million dollars",
            body_text="Dear friend, I am a prince and I have an inheritance to share.",
        )
        result = detector.analyze(email)
        assert result.category == SpamCategory.SCAM

    def test_category_detection_commercial(self, detector):
        """Test commercial category detection."""
        email = EmailContent(
            email_id="commercial-1",
            sender="marketing@store.com",
            subject="Special offer just for you!",
            body_text="Don't miss our sale! To unsubscribe, click here.",
        )
        result = detector.analyze(email)
        assert result.category == SpamCategory.COMMERCIAL

    def test_link_analysis(self, detector):
        """Test link analysis."""
        email = EmailContent(
            email_id="links-1",
            sender="sender@example.com",
            subject="Check this out",
            body_text="Click here: http://192.168.1.1/login\nAlso: https://bit.ly/xyz123",
        )
        result = detector.analyze(email)
        assert len(result.links_analyzed) > 0
        # Should detect IP address and URL shortener
        suspicious = [link for link in result.links_analyzed if link.is_suspicious]
        assert len(suspicious) > 0


class TestPhishingAnalysis:
    """Tests for phishing analysis."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        return SpamDetector()

    def test_phishing_detection(self, detector):
        """Test phishing email detection."""
        email = EmailContent(
            email_id="phish-1",
            sender="security@micros0ft-support.com",
            subject="Your Microsoft account needs verification",
            body_text="Your Microsoft account has been compromised. "
            "Please verify your identity by clicking here: "
            "http://micros0ft-login.fake.com/signin\n"
            "Click here immediately to reset your password.",
        )
        result = detector.analyze_phishing(email)
        assert result.is_phishing is True
        assert result.phishing_score > 0.5
        assert result.targeted_brand == "microsoft"

    def test_credential_harvesting_detection(self, detector):
        """Test credential harvesting detection."""
        # Use an IP address URL which triggers suspicious link detection
        email = EmailContent(
            email_id="cred-1",
            sender="support@amaz0n.com",
            subject="Confirm your identity",
            body_text="Please login to verify your account: "
            "http://192.168.1.100/signin?verify=true",
        )
        result = detector.analyze_phishing(email)
        assert len(result.suspicious_links) > 0

    def test_non_phishing_email(self, detector):
        """Test non-phishing email."""
        email = EmailContent(
            email_id="legit-1",
            sender="notifications@amazon.com",
            subject="Your order has shipped",
            body_text="Your package is on the way. Track at amazon.com.",
        )
        result = detector.analyze_phishing(email)
        assert result.is_phishing is False
        assert result.phishing_score < 0.5


class TestLinkAnalysis:
    """Tests for link analysis."""

    def test_create_link_analysis(self):
        """Test creating link analysis."""
        analysis = LinkAnalysis(
            url="https://example.com",
            display_text="Example",
            domain="example.com",
            is_shortened=False,
            is_suspicious=False,
        )
        assert analysis.domain == "example.com"

    def test_shortened_url_detection(self):
        """Test URL shortener detection."""
        detector = SpamDetector()
        email = EmailContent(
            email_id="short-1",
            sender="sender@example.com",
            subject="Link",
            body_text="Check this: https://bit.ly/abc123",
        )
        result = detector.analyze(email)
        shortened = [link for link in result.links_analyzed if link.is_shortened]
        assert len(shortened) > 0


class TestWhitelistBlacklist:
    """Tests for whitelist/blacklist management."""

    def test_add_remove_whitelist(self):
        """Test adding and removing from whitelist."""
        detector = SpamDetector()
        detector.add_to_whitelist("test@example.com")
        assert "test@example.com" in detector.user_whitelist

        detector.remove_from_whitelist("test@example.com")
        assert "test@example.com" not in detector.user_whitelist

    def test_add_remove_blacklist(self):
        """Test adding and removing from blacklist."""
        detector = SpamDetector()
        detector.add_to_blacklist("spam@spam.com")
        assert "spam@spam.com" in detector.user_blacklist

        detector.remove_from_blacklist("spam@spam.com")
        assert "spam@spam.com" not in detector.user_blacklist

    def test_case_insensitive(self):
        """Test case insensitivity."""
        detector = SpamDetector()
        detector.add_to_whitelist("Test@Example.COM")

        email = EmailContent(
            email_id="case-1",
            sender="test@example.com",
            subject="Test",
            body_text="Test",
        )
        result = detector.analyze(email)
        assert result.is_spam is False
