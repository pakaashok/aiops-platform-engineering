"""
Unit tests for the IntentClassifier.
Fast, isolated, no network calls, no external dependencies.
"""

import pytest
from app.intent_classifier import IntentClassifier, get_classifier


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_classifier():
    """Fresh trained classifier for each test."""
    clf = IntentClassifier()
    clf.train()
    return clf


@pytest.fixture
def untrained_classifier():
    """Untrained classifier for negative testing."""
    return IntentClassifier()


# ── Training Tests ────────────────────────────────────────────────────────────

class TestClassifierTraining:

    def test_classifier_starts_untrained(self, untrained_classifier):
        """A new classifier should not be trained."""
        assert untrained_classifier.is_trained is False

    def test_classifier_is_trained_after_train(self, trained_classifier):
        """After calling train(), is_trained should be True."""
        assert trained_classifier.is_trained is True

    def test_predict_raises_before_training(self, untrained_classifier):
        """Calling predict() before train() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="must be trained"):
            untrained_classifier.predict("check cpu usage")


# ── Prediction Tests ──────────────────────────────────────────────────────────

class TestClassifierPredictions:

    def test_predict_returns_tuple(self, trained_classifier):
        """predict() should return a (label, confidence) tuple."""
        result = trained_classifier.predict("show me cpu usage")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_confidence_is_float_between_0_and_1(self, trained_classifier):
        """Confidence score must be a valid probability."""
        _, confidence = trained_classifier.predict("memory is high")
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_label_is_valid_intent(self, trained_classifier):
        """Predicted label must be one of the known intent categories."""
        valid_intents = {
            "metrics", "remediation", "logs", "health", "unknown"
        }
        intent, _ = trained_classifier.predict("check disk space")
        assert intent in valid_intents

    @pytest.mark.parametrize("query,expected_intent", [
        ("show me cpu usage",              "metrics"),
        ("check memory utilization",       "metrics"),
        ("restart the pod",                "remediation"),
        ("rollback to previous version",   "remediation"),
        ("show me recent errors in logs",  "logs"),
        ("is the service healthy",         "health"),
        ("check pod status",               "health"),
    ])
    def test_known_queries_predict_correct_intent(
        self, trained_classifier, query, expected_intent
    ):
        """
        Parametrized test — each (query, intent) pair is a
        separate test case. Core accuracy regression tests.
        """
        intent, confidence = trained_classifier.predict(query)
        assert intent == expected_intent, (
            f"Query '{query}' predicted as '{intent}' "
            f"but expected '{expected_intent}' "
            f"(confidence: {confidence:.3f})"
        )


# ── Edge Case Tests ───────────────────────────────────────────────────────────

class TestClassifierEdgeCases:

    def test_empty_string_returns_unknown(self, trained_classifier):
        """Empty string should return unknown intent."""
        intent, confidence = trained_classifier.predict("")
        assert intent == "unknown"
        assert confidence == 0.0

    def test_whitespace_only_returns_unknown(self, trained_classifier):
        """Whitespace-only string should return unknown intent."""
        intent, _ = trained_classifier.predict("   ")
        assert intent == "unknown"

    def test_very_long_query(self, trained_classifier):
        """Very long queries should not crash the classifier."""
        long_query = "check cpu memory disk network " * 100
        intent, confidence = trained_classifier.predict(long_query)
        assert isinstance(intent, str)
        assert isinstance(confidence, float)

    def test_special_characters(self, trained_classifier):
        """Queries with special characters should not crash."""
        intent, confidence = trained_classifier.predict(
            "ERROR: pod/nginx-7d4b9c crashed @2024-01-01"
        )
        assert isinstance(intent, str)
        assert isinstance(confidence, float)

    def test_predict_batch_returns_correct_length(self, trained_classifier):
        """Batch prediction should return same count as inputs."""
        queries = ["cpu usage", "restart pod", "check logs"]
        results = trained_classifier.predict_batch(queries)
        assert len(results) == len(queries)

    def test_singleton_returns_same_instance(self):
        """get_classifier() should always return the same instance."""
        clf1 = get_classifier()
        clf2 = get_classifier()
        assert clf1 is clf2


# ── Confidence Threshold Tests ────────────────────────────────────────────────

class TestConfidenceThreshold:

    def test_high_threshold_returns_unknown(self):
        """Very high threshold should result in unknown predictions."""
        strict_clf = IntentClassifier(confidence_threshold=0.99)
        strict_clf.train()
        intent, _ = strict_clf.predict("something vague here")
        assert intent == "unknown"

    def test_clear_query_has_nonzero_confidence(self, trained_classifier):
        """A clear operational query should have confidence above zero."""
        _, confidence = trained_classifier.predict("cpu usage")
        assert confidence > 0.0
