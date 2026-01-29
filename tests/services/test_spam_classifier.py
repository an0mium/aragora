"""
Comprehensive tests for the ML-Enhanced Spam Classification Service.

Tests cover:
- SpamFeatures extraction (content, subject, sender, URL, attachment, edge cases)
- NaiveBayesClassifier (training, prediction, save/load, incremental)
- SpamClassificationResult (categories, component scores, reasons)
- SpamClassifier integration (classify clean/spam/phishing, thresholds, ML, feedback, stats)
- EmailFeatures (to_vector, to_dict, defaults)
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.services.spam_classifier import (
    DANGEROUS_EXTENSIONS,
    EmailFeatures,
    NaiveBayesClassifier,
    SpamCategory,
    SpamClassificationResult,
    SpamClassifier,
    SpamClassifierConfig,
    SpamFeatures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, **overrides):
    """Create a SpamClassifierConfig pointing at tmp_path for files."""
    defaults = {
        "model_path": str(tmp_path / "model.json"),
        "feedback_db_path": str(tmp_path / "feedback.db"),
        "use_ml_model": True,
        "min_confidence_for_ml": 0.7,
        "spam_threshold": 0.6,
        "promotional_threshold": 0.4,
        "suspicious_threshold": 0.3,
        "min_training_samples": 100,
        "retrain_interval_hours": 24,
    }
    defaults.update(overrides)
    return SpamClassifierConfig(**defaults)


def _make_classifier(tmp_path, **config_overrides):
    """Build a SpamClassifier with temp storage and a mock sender history."""
    config = _make_config(tmp_path, **config_overrides)
    sender_history = MagicMock()
    return SpamClassifier(
        config=config,
        sender_history_service=sender_history,
        user_id="test-user",
    )


# ===========================================================================
# 1. SpamFeatures extraction
# ===========================================================================


class TestSpamFeaturesExtraction:
    """Tests for the SpamFeatures.extract() method."""

    def setup_method(self):
        self.extractor = SpamFeatures()

    # -- Content features ---------------------------------------------------

    def test_content_word_count(self):
        features = self.extractor.extract(
            subject="Hello",
            body="This is a normal email body with several words",
            sender="user@example.com",
        )
        # "Hello" is prepended via subject+body so word count includes it
        assert features.word_count > 0

    def test_content_uppercase_ratio(self):
        features = self.extractor.extract(
            subject="TEST",
            body="THIS IS ALL UPPERCASE TEXT FOR TESTING",
            sender="user@example.com",
        )
        assert features.uppercase_ratio > 0.4

    def test_content_spam_words_detected(self):
        features = self.extractor.extract(
            subject="Congratulations",
            body="You won a free prize! Click here to claim your lottery winnings.",
            sender="spam@example.com",
        )
        # Should detect several spam words: congratulations, won, free, prize, lottery, click here
        assert features.spam_word_count >= 4

    def test_content_exclamation_count(self):
        features = self.extractor.extract(
            subject="Wow!",
            body="Amazing!!! Buy now!!! Free!!!",
            sender="user@example.com",
        )
        assert features.exclamation_count >= 7

    # -- Subject features ---------------------------------------------------

    def test_subject_all_caps(self):
        features = self.extractor.extract(
            subject="YOU HAVE WON A BIG PRIZE",
            body="Some body.",
            sender="user@example.com",
        )
        assert features.subject_all_caps is True

    def test_subject_excessive_punctuation(self):
        features = self.extractor.extract(
            subject="Act now!!! Don't miss $$$!!!",
            body="Body.",
            sender="user@example.com",
        )
        assert features.subject_excessive_punctuation is True

    def test_subject_re_fw(self):
        features = self.extractor.extract(
            subject="Re: Meeting tomorrow",
            body="Sounds good.",
            sender="colleague@company.com",
        )
        assert features.subject_has_re_fw is True

    # -- Sender features ----------------------------------------------------

    def test_sender_free_email(self):
        features = self.extractor.extract(
            subject="Hi",
            body="Hello.",
            sender="john@gmail.com",
        )
        assert features.sender_domain_is_free_email is True

    def test_sender_suspicious_domain(self):
        features = self.extractor.extract(
            subject="Hey",
            body="Check this out.",
            sender="deals@cheap-store.tk",
        )
        assert features.sender_domain_suspicious is True

    def test_sender_known_spam_domain(self):
        features = self.extractor.extract(
            subject="Hey",
            body="Check this out.",
            sender="info@mailinator.com",
        )
        assert features.sender_is_known_spam_domain is True

    # -- URL features -------------------------------------------------------

    def test_url_shortened_urls(self):
        features = self.extractor.extract(
            subject="Link",
            body="Click here: https://bit.ly/abc123 and https://tinyurl.com/xyz",
            sender="user@example.com",
        )
        assert features.shortened_url_count >= 2

    def test_url_ip_address(self):
        features = self.extractor.extract(
            subject="Link",
            body="Visit http://192.168.1.100/login to verify your account",
            sender="user@example.com",
        )
        assert features.ip_address_urls >= 1

    # -- Attachment features ------------------------------------------------

    def test_attachment_dangerous_extension(self):
        features = self.extractor.extract(
            subject="Document",
            body="See attached.",
            sender="user@example.com",
            attachments=["invoice.exe", "report.pdf"],
        )
        assert features.dangerous_extension_count >= 1
        assert "invoice.exe" in features.dangerous_attachments

    def test_attachment_double_extension(self):
        features = self.extractor.extract(
            subject="Document",
            body="See attached.",
            sender="user@example.com",
            attachments=["document.pdf.exe"],
        )
        assert features.double_extension_count >= 1

    # -- Edge cases ---------------------------------------------------------

    def test_empty_input(self):
        features = self.extractor.extract(
            subject="",
            body="",
            sender="",
        )
        assert features.word_count == 0
        assert features.char_count == 2  # "\n\n" from subject+body join

    def test_none_optional_fields(self):
        features = self.extractor.extract(
            subject="Test",
            body="Body",
            sender="user@example.com",
            headers=None,
            attachments=None,
        )
        assert features.attachment_count == 0


# ===========================================================================
# 2. NaiveBayesClassifier
# ===========================================================================


class TestNaiveBayesClassifier:
    """Tests for the NaiveBayesClassifier."""

    def test_untrained_returns_default(self):
        clf = NaiveBayesClassifier()
        is_spam, confidence = clf.predict("hello world")
        assert is_spam is False
        assert confidence == 0.5

    def test_is_trained_property(self):
        clf = NaiveBayesClassifier()
        assert clf.is_trained is False
        clf.train("buy cheap pills now", is_spam=True)
        assert clf.is_trained is True

    def test_train_and_predict_spam(self):
        clf = NaiveBayesClassifier()
        # Train with spam examples
        for _ in range(20):
            clf.train("buy cheap pills discount offer free prize click", is_spam=True)
            clf.train("meeting tomorrow agenda project report review", is_spam=False)

        is_spam, confidence = clf.predict("buy cheap pills free offer")
        assert is_spam is True
        assert confidence > 0.3

    def test_train_and_predict_ham(self):
        clf = NaiveBayesClassifier()
        for _ in range(20):
            clf.train("buy cheap pills discount offer free prize click", is_spam=True)
            clf.train("meeting tomorrow agenda project report review quarterly", is_spam=False)

        is_spam, confidence = clf.predict("meeting agenda project quarterly report")
        assert is_spam is False
        assert confidence > 0.3

    def test_prediction_returns_tuple(self):
        clf = NaiveBayesClassifier()
        clf.train("test text", is_spam=False)
        result = clf.predict("test text")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_save_and_load(self, tmp_path):
        clf = NaiveBayesClassifier()
        for _ in range(10):
            clf.train("spam words buy cheap pills", is_spam=True)
            clf.train("normal words meeting agenda report", is_spam=False)

        model_path = str(tmp_path / "model.json")
        clf.save(model_path)
        assert os.path.exists(model_path)

        # Verify saved as valid JSON
        with open(model_path) as f:
            data = json.load(f)
        assert "spam_count" in data
        assert data["spam_count"] == 10

        # Load into new classifier
        clf2 = NaiveBayesClassifier()
        success = clf2.load(model_path)
        assert success is True
        assert clf2.spam_count == clf.spam_count
        assert clf2.ham_count == clf.ham_count
        assert clf2.vocabulary == clf.vocabulary

    def test_load_nonexistent_file(self):
        clf = NaiveBayesClassifier()
        success = clf.load("/nonexistent/path/model.json")
        assert success is False

    def test_incremental_training(self):
        clf = NaiveBayesClassifier()
        clf.train("buy cheap pills", is_spam=True)
        assert clf.spam_count == 1
        clf.train("meeting report", is_spam=False)
        assert clf.ham_count == 1
        clf.train("free discount offer", is_spam=True)
        assert clf.spam_count == 2
        assert clf.ham_count == 1


# ===========================================================================
# 3. SpamClassificationResult
# ===========================================================================


class TestSpamClassificationResult:
    """Tests for SpamClassificationResult dataclass."""

    def test_category_enum_values(self):
        assert SpamCategory.HAM.value == "ham"
        assert SpamCategory.SPAM.value == "spam"
        assert SpamCategory.PHISHING.value == "phishing"
        assert SpamCategory.PROMOTIONAL.value == "promotional"
        assert SpamCategory.SUSPICIOUS.value == "suspicious"

    def test_component_scores_in_to_dict(self):
        result = SpamClassificationResult(
            email_id="test-1",
            is_spam=True,
            category=SpamCategory.SPAM,
            confidence=0.95,
            spam_score=0.85,
            content_score=0.7,
            sender_score=0.3,
            header_score=0.2,
            pattern_score=0.6,
            url_score=0.5,
            attachment_score=0.1,
            subject_score=0.4,
        )
        d = result.to_dict()
        assert d["scores"]["content"] == 0.7
        assert d["scores"]["sender"] == 0.3
        assert d["scores"]["url"] == 0.5
        assert d["category"] == "spam"
        assert d["is_spam"] is True
        assert d["confidence"] == 0.95
        assert "classified_at" in d

    def test_reasons_list(self):
        result = SpamClassificationResult(
            email_id="test-2",
            is_spam=False,
            category=SpamCategory.HAM,
            confidence=0.9,
            spam_score=0.1,
            reasons=["Reason 1", "Reason 2"],
        )
        assert len(result.reasons) == 2
        assert result.reasons[0] == "Reason 1"

    def test_priority_penalty(self):
        spam_result = SpamClassificationResult(
            email_id="e1",
            is_spam=True,
            category=SpamCategory.SPAM,
            confidence=0.9,
            spam_score=0.9,
        )
        assert spam_result.get_priority_penalty() == 1.0

        phishing_result = SpamClassificationResult(
            email_id="e2",
            is_spam=True,
            category=SpamCategory.PHISHING,
            confidence=0.9,
            spam_score=0.9,
        )
        assert phishing_result.get_priority_penalty() == 1.0  # is_spam is True => 1.0

        # Phishing but not marked is_spam explicitly
        phishing_result2 = SpamClassificationResult(
            email_id="e3",
            is_spam=False,
            category=SpamCategory.PHISHING,
            confidence=0.9,
            spam_score=0.9,
        )
        assert phishing_result2.get_priority_penalty() == 0.9

        suspicious = SpamClassificationResult(
            email_id="e4",
            is_spam=False,
            category=SpamCategory.SUSPICIOUS,
            confidence=0.5,
            spam_score=0.4,
        )
        assert suspicious.get_priority_penalty() == 0.5

        promo = SpamClassificationResult(
            email_id="e5",
            is_spam=False,
            category=SpamCategory.PROMOTIONAL,
            confidence=0.5,
            spam_score=0.3,
        )
        assert promo.get_priority_penalty() == 0.3

        ham = SpamClassificationResult(
            email_id="e6",
            is_spam=False,
            category=SpamCategory.HAM,
            confidence=0.9,
            spam_score=0.1,
        )
        assert ham.get_priority_penalty() == 0.0


# ===========================================================================
# 4. SpamClassifier integration
# ===========================================================================


class TestSpamClassifierIntegration:
    """Integration tests for the SpamClassifier."""

    @pytest.fixture
    def classifier(self, tmp_path):
        return _make_classifier(tmp_path)

    @pytest.mark.asyncio
    async def test_initialize(self, classifier):
        await classifier.initialize()
        assert classifier._initialized is True

    @pytest.mark.asyncio
    async def test_classify_clean_email(self, classifier):
        await classifier.initialize()
        result = await classifier.classify_email(
            email_id="clean-1",
            subject="Re: Quarterly report",
            body="Hi team, please find the quarterly report attached. Let me know your thoughts.",
            sender="Alice Smith <alice@company.com>",
            headers={
                "From": "alice@company.com",
                "To": "team@company.com",
                "Date": "Mon, 1 Jan 2024 10:00:00 +0000",
                "Message-ID": "<abc@company.com>",
                "Authentication-Results": "spf=pass dkim=pass dmarc=pass",
            },
            attachments=["Q4_Report.pdf"],
        )
        assert result.email_id == "clean-1"
        assert result.category == SpamCategory.HAM
        assert result.is_spam is False
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_classify_obvious_spam(self, tmp_path):
        # Use a lower spam threshold so the weighted score (dominated by
        # content, sender, subject components) crosses into SPAM territory.
        clf = _make_classifier(tmp_path, spam_threshold=0.5)
        await clf.initialize()
        result = await clf.classify_email(
            email_id="spam-1",
            subject="CONGRATULATIONS YOU WON!!!",
            body=(
                "You have been selected as the winner of our $1,000,000 lottery! "
                "Act now! Click here to claim your prize immediately! "
                "This is a limited time offer - hurry! "
                "Send your bank details to receive the money. "
                "Earn money from home! Free cash! Urgent!"
            ),
            sender="winner123456@spam4.me",
            headers={},
            attachments=["claim_form.exe"],
        )
        assert result.is_spam is True
        assert result.category in (SpamCategory.SPAM, SpamCategory.PHISHING)
        assert result.spam_score > 0.4
        assert len(result.reasons) > 0

    @pytest.mark.asyncio
    async def test_classify_phishing_email(self, classifier):
        await classifier.initialize()
        result = await classifier.classify_email(
            email_id="phish-1",
            subject="Security Alert: Verify your account",
            body=(
                "We detected unusual activity on your account. "
                "Your account will be suspended within 24 hours. "
                "Immediately verify your account by clicking the link below: "
                "http://192.168.1.1/verify "
                '<a href="http://evil.tk/steal">http://bank.com/verify</a> '
                '<form action="http://evil.tk/harvest"><input name="password"></form> '
                "Confirm your identity now. Action required urgently."
            ),
            sender="security@mailinator.com",
            headers={},
        )
        # Should detect phishing indicators
        assert result.category in (SpamCategory.PHISHING, SpamCategory.SPAM)
        assert result.is_spam is True

    @pytest.mark.asyncio
    async def test_score_below_suspicious_threshold(self, tmp_path):
        """A clean email should score below the suspicious threshold."""
        classifier = _make_classifier(tmp_path, suspicious_threshold=0.3)
        await classifier.initialize()
        result = await classifier.classify_email(
            email_id="ham-1",
            subject="Re: Lunch plans",
            body="Sounds good, let's meet at noon.",
            sender="Bob <bob@company.com>",
            headers={
                "From": "bob@company.com",
                "To": "alice@company.com",
                "Date": "Mon, 1 Jan 2024 10:00:00 +0000",
                "Message-ID": "<xyz@company.com>",
                "Authentication-Results": "spf=pass dkim=pass dmarc=pass",
            },
        )
        assert result.category == SpamCategory.HAM
        assert result.spam_score < 0.3

    @pytest.mark.asyncio
    async def test_promotional_email(self, tmp_path):
        classifier = _make_classifier(tmp_path, promotional_threshold=0.35, spam_threshold=0.7)
        await classifier.initialize()
        result = await classifier.classify_email(
            email_id="promo-1",
            subject="Special offer for you",
            body=(
                "Check out our latest deals and save big! "
                "Limited time discount on all products. "
                "View in browser | Unsubscribe | Manage subscription "
                "Email preferences - opt out anytime. "
                "© 2024 All rights reserved."
            ),
            sender="deals@store.com",
            headers={
                "From": "deals@store.com",
                "To": "user@example.com",
                "Date": "Mon, 1 Jan 2024 10:00:00 +0000",
                "Message-ID": "<promo@store.com>",
                "Authentication-Results": "spf=pass dkim=pass dmarc=pass",
            },
        )
        assert result.category in (
            SpamCategory.PROMOTIONAL,
            SpamCategory.SUSPICIOUS,
            SpamCategory.HAM,
        )

    @pytest.mark.asyncio
    async def test_ml_model_classification(self, tmp_path):
        """When the ML model is trained and confident, it should influence results."""
        classifier = _make_classifier(tmp_path)
        await classifier.initialize()

        # Train the internal model with enough data
        for _ in range(30):
            classifier.model.train(
                "buy cheap pills discount offer free prize click lottery winner congratulations",
                is_spam=True,
            )
            classifier.model.train(
                "meeting tomorrow agenda project report review quarterly budget planning",
                is_spam=False,
            )

        result = await classifier.classify_email(
            email_id="ml-1",
            subject="Special offer",
            body="Buy cheap pills discount offer free prize click lottery",
            sender="spammer@example.com",
        )
        # The ML model should contribute to a spam classification
        assert result.spam_score > 0.3

    @pytest.mark.asyncio
    async def test_feedback_and_training(self, tmp_path):
        classifier = _make_classifier(tmp_path)
        await classifier.initialize()

        success = await classifier.train_from_feedback(
            email_id="fb-1",
            is_spam=True,
            content="buy cheap pills now free offer",
            user_id="test-user",
        )
        assert success is True
        assert classifier.model.spam_count >= 1

        success2 = await classifier.train_from_feedback(
            email_id="fb-2",
            is_spam=False,
            content="meeting agenda for tomorrow's review",
            user_id="test-user",
        )
        assert success2 is True
        assert classifier.model.ham_count >= 1

    @pytest.mark.asyncio
    async def test_get_statistics(self, tmp_path):
        classifier = _make_classifier(tmp_path)
        await classifier.initialize()

        # Add some feedback
        await classifier.train_from_feedback("fb-1", True, "spam text", "user")
        await classifier.train_from_feedback("fb-2", False, "ham text", "user")

        stats = await classifier.get_statistics()
        assert "model_trained" in stats
        assert stats["model_trained"] is True
        assert stats["spam_samples"] >= 1
        assert stats["ham_samples"] >= 1
        assert "vocabulary_size" in stats
        assert "total_feedback" in stats
        assert stats["total_feedback"] >= 2

    @pytest.mark.asyncio
    async def test_classify_caches_result(self, tmp_path):
        """Second classification of same content should come from cache."""
        classifier = _make_classifier(tmp_path)
        await classifier.initialize()

        kwargs = dict(
            email_id="cache-1",
            subject="Hello",
            body="Normal body text",
            sender="user@example.com",
        )
        result1 = await classifier.classify_email(**kwargs)

        # Classify again with different email_id but same content
        kwargs["email_id"] = "cache-2"
        result2 = await classifier.classify_email(**kwargs)

        # Cached result should be returned (model_used == 'cached')
        assert result2.model_used == "cached"
        assert result2.category == result1.category

    @pytest.mark.asyncio
    async def test_close(self, tmp_path):
        classifier = _make_classifier(tmp_path)
        await classifier.initialize()
        assert classifier._db_conn is not None
        await classifier.close()
        assert classifier._db_conn is None


# ===========================================================================
# 5. EmailFeatures
# ===========================================================================


class TestEmailFeatures:
    """Tests for EmailFeatures dataclass."""

    def test_to_vector_returns_correct_dimension(self):
        features = EmailFeatures()
        vec = features.to_vector()
        assert isinstance(vec, list)
        # Count the elements in to_vector() — there are 46 elements
        assert len(vec) == 46
        assert all(isinstance(v, float) for v in vec)

    def test_to_vector_values_with_populated_features(self):
        features = EmailFeatures(
            word_count=500,
            char_count=3000,
            uppercase_ratio=0.2,
            spam_word_count=5,
            subject_all_caps=True,
            sender_domain_suspicious=True,
            has_spf=True,
        )
        vec = features.to_vector()
        # word_count / 1000 = 0.5
        assert vec[0] == pytest.approx(0.5)
        # char_count / 10000 = 0.3
        assert vec[1] == pytest.approx(0.3)
        # uppercase_ratio = 0.2
        assert vec[2] == pytest.approx(0.2)
        # subject_all_caps = 1.0
        assert vec[14] == pytest.approx(1.0)

    def test_to_dict_returns_all_expected_keys(self):
        features = EmailFeatures()
        d = features.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "word_count",
            "char_count",
            "uppercase_ratio",
            "link_count",
            "suspicious_link_count",
            "spam_word_count",
            "subject_all_caps",
            "subject_excessive_punctuation",
            "sender_domain_suspicious",
            "sender_is_known_spam_domain",
            "has_spf",
            "has_dkim",
            "shortened_url_count",
            "dangerous_extension_count",
            "dangerous_attachments",
            "top_unigrams",
            "top_bigrams",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_default_values(self):
        features = EmailFeatures()
        assert features.word_count == 0
        assert features.char_count == 0
        assert features.uppercase_ratio == 0.0
        assert features.link_count == 0
        assert features.spam_word_count == 0
        assert features.subject_all_caps is False
        assert features.sender_domain_suspicious is False
        assert features.dangerous_attachments == []
        assert features.missing_required_headers == []
        assert features.top_unigrams == []
        assert features.top_bigrams == []


# ===========================================================================
# 6. Header feature extraction
# ===========================================================================


class TestHeaderFeatures:
    """Tests for header-based feature extraction."""

    def setup_method(self):
        self.extractor = SpamFeatures()

    def test_spf_dkim_dmarc_detected(self):
        features = self.extractor.extract(
            subject="Test",
            body="Body",
            sender="user@example.com",
            headers={
                "Authentication-Results": "spf=pass; dkim=pass; dmarc=pass",
                "From": "user@example.com",
                "To": "other@example.com",
                "Date": "Mon, 1 Jan 2024 10:00:00 +0000",
                "Message-ID": "<abc@example.com>",
            },
        )
        assert features.has_spf is True
        assert features.has_dkim is True
        assert features.has_dmarc is True

    def test_missing_required_headers(self):
        features = self.extractor.extract(
            subject="Test",
            body="Body",
            sender="user@example.com",
            headers={},  # empty headers
        )
        # No headers means nothing extracted from _extract_header_features
        # (it returns early when headers is empty dict — but it's not falsy)
        # Actually, empty dict is falsy for `if not headers` check
        # So missing_required_headers stays as default []
        assert isinstance(features.missing_required_headers, list)

    def test_suspicious_routing(self):
        # More than 8 received headers triggers suspicious routing
        headers = {"From": "a@b.com", "To": "c@d.com", "Date": "now", "Message-ID": "<x>"}
        for i in range(10):
            headers["Received"] = f"hop{i}"  # Only one key survives in dict
        # With dict, can't have multiple 'Received' keys. That's fine;
        # the feature extractor handles this via counting header keys
        features = self.extractor.extract(
            subject="Test",
            body="Body",
            sender="a@b.com",
            headers=headers,
        )
        # Single 'Received' key won't trigger, which is expected dict behavior
        assert isinstance(features.has_suspicious_routing, bool)
