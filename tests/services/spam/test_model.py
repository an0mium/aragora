from __future__ import annotations

import json

from aragora.services.spam.model import NaiveBayesClassifier


def _train_classifier() -> NaiveBayesClassifier:
    classifier = NaiveBayesClassifier()
    spam_examples = [
        "win free cash now limited offer click here",
        "cheap meds free prize urgent claim reward",
    ]
    ham_examples = [
        "team meeting notes and project update",
        "lunch tomorrow with design review agenda",
    ]
    for _ in range(12):
        for example in spam_examples:
            classifier.train(example, is_spam=True)
        for example in ham_examples:
            classifier.train(example, is_spam=False)
    return classifier


def test_initialization_uses_default_state() -> None:
    classifier = NaiveBayesClassifier()

    assert classifier.spam_count == 0
    assert classifier.ham_count == 0
    assert classifier.vocabulary == set()
    assert classifier.is_trained is False


def test_predict_returns_default_for_untrained_classifier() -> None:
    classifier = NaiveBayesClassifier()

    is_spam, confidence = classifier.predict("hello world")

    assert is_spam is False
    assert confidence == 0.5


def test_predict_flags_obvious_spam_after_training() -> None:
    classifier = _train_classifier()

    is_spam, confidence = classifier.predict("free cash prize click now")

    assert is_spam is True
    assert confidence > 0.3


def test_predict_keeps_regular_content_below_spam_threshold() -> None:
    classifier = _train_classifier()

    is_spam, confidence = classifier.predict("project update and meeting notes")

    assert is_spam is False
    assert 0.0 <= confidence <= 1.0


def test_tokenize_normalizes_case_and_special_characters() -> None:
    classifier = NaiveBayesClassifier()

    tokens = classifier._tokenize("WIN big $$$ now!!! Email SUPPORT@example.com today.")

    assert tokens == ["win", "big", "now", "email", "support", "example", "com", "today"]


def test_tokenize_handles_empty_and_symbol_only_text() -> None:
    classifier = NaiveBayesClassifier()

    assert classifier._tokenize("") == []
    assert classifier._tokenize("!!! $$$ ...") == []


def test_predict_handles_very_long_text() -> None:
    classifier = _train_classifier()
    long_text = " ".join(["offer"] * 2000)

    is_spam, confidence = classifier.predict(long_text)

    assert is_spam is True
    assert confidence >= 0.0


def test_save_and_load_round_trip_preserves_model_state(tmp_path) -> None:
    classifier = _train_classifier()
    model_path = tmp_path / "spam-model.json"

    classifier.save(str(model_path))

    restored = NaiveBayesClassifier()
    assert restored.load(str(model_path)) is True
    assert restored.spam_count == classifier.spam_count
    assert restored.ham_count == classifier.ham_count
    assert restored.word_spam_counts == classifier.word_spam_counts
    assert restored.word_ham_counts == classifier.word_ham_counts
    assert restored.vocabulary == classifier.vocabulary


def test_load_returns_false_for_invalid_json(tmp_path) -> None:
    model_path = tmp_path / "invalid.json"
    model_path.write_text("{not json", encoding="utf-8")

    classifier = NaiveBayesClassifier()

    assert classifier.load(str(model_path)) is False


def test_load_returns_false_for_invalid_model_data(tmp_path) -> None:
    model_path = tmp_path / "invalid-model.json"
    model_path.write_text(
        json.dumps(
            {
                "word_spam_counts": {"free": 3},
                "word_ham_counts": {"project": 2},
                "ham_count": 2,
                "vocabulary": ["free", "project"],
            }
        ),
        encoding="utf-8",
    )

    classifier = NaiveBayesClassifier()

    assert classifier.load(str(model_path)) is False
