"""
Intent classifier for AIOps assistant.
Classifies incoming queries into operational intent categories.
This is the ML component we will write tests against.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Tuple


TRAINING_DATA = [
    ("show me cpu usage", "metrics"),
    ("what is the memory utilization", "metrics"),
    ("check disk space", "metrics"),
    ("pod cpu is high", "metrics"),
    ("memory is running out", "metrics"),
    ("show network traffic", "metrics"),
    ("restart the pod", "remediation"),
    ("scale up the deployment", "remediation"),
    ("rollback to previous version", "remediation"),
    ("delete the failed pod", "remediation"),
    ("increase replicas", "remediation"),
    ("error in application logs", "logs"),
    ("show me recent errors", "logs"),
    ("tail the logs", "logs"),
    ("find exception in logs", "logs"),
    ("check application logs", "logs"),
    ("is the service healthy", "health"),
    ("check pod status", "health"),
    ("is the deployment ready", "health"),
    ("service is down", "health"),
    ("ping the endpoint", "health"),
]

LABELS = [label for _, label in TRAINING_DATA]
TEXTS  = [text  for text, _ in TRAINING_DATA]

INTENT_CATEGORIES = ["metrics", "remediation", "logs", "health", "unknown"]


class IntentClassifier:
    """
    A lightweight TF-IDF + Naive Bayes classifier for operational intents.
    Designed to be fast, testable, and replaceable with an LLM later.
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=1000
            )),
            ("clf", MultinomialNB(alpha=0.1))
        ])
        self._is_trained = False

    def train(self) -> None:
        """Train the classifier on built-in training data."""
        self.model.fit(TEXTS, LABELS)
        self._is_trained = True

    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict the intent of a query.

        Args:
            query: The user's natural language query

        Returns:
            Tuple of (intent_label, confidence_score)

        Raises:
            RuntimeError: If model has not been trained yet
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model must be trained before calling predict(). "
                "Call train() first."
            )

        if not query or not query.strip():
            return ("unknown", 0.0)

        proba    = self.model.predict_proba([query.strip()])[0]
        classes  = self.model.classes_
        best_idx = np.argmax(proba)

        best_label      = classes[best_idx]
        best_confidence = float(proba[best_idx])

        if best_confidence < self.confidence_threshold:
            return ("unknown", best_confidence)

        return (best_label, best_confidence)

    def predict_batch(self, queries: list) -> list:
        """Predict intents for multiple queries at once."""
        return [self.predict(q) for q in queries]

    @property
    def is_trained(self) -> bool:
        return self._is_trained


_classifier_instance = None


def get_classifier() -> IntentClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
        _classifier_instance.train()
    return _classifier_instance
