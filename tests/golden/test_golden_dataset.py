"""
Golden dataset tests for the AIOps intent classifier.

Golden dataset testing ensures model changes do not cause
regressions on a known-good set of labeled examples.
If accuracy drops below the threshold the CI pipeline fails.
"""

import json
import pytest
from pathlib import Path
from app.intent_classifier import IntentClassifier


DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def golden_data():
    """Load the golden dataset once for all tests in this module."""
    with open(DATASET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def trained_classifier():
    """Train classifier once for all golden dataset tests."""
    clf = IntentClassifier()
    clf.train()
    return clf


# ── Golden Dataset Tests ──────────────────────────────────────────────────────

class TestGoldenDataset:

    def test_dataset_file_exists(self):
        """Golden dataset file must exist in the repository."""
        assert DATASET_PATH.exists(), (
            f"Golden dataset not found at {DATASET_PATH}"
        )

    def test_dataset_has_test_cases(self, golden_data):
        """Golden dataset must contain at least one test case."""
        assert len(golden_data["test_cases"]) > 0

    def test_overall_accuracy_meets_threshold(
        self, trained_classifier, golden_data
    ):
        """
        Classifier accuracy on the golden dataset must meet
        the minimum threshold. This is our ML regression gate.
        If this fails a code change degraded model performance.
        """
        test_cases        = golden_data["test_cases"]
        minimum_accuracy  = golden_data["minimum_accuracy"]

        correct  = 0
        failures = []

        for case in test_cases:
            query     = case["query"]
            expected  = case["expected_intent"]
            predicted, confidence = trained_classifier.predict(query)

            if predicted == expected:
                correct += 1
            else:
                failures.append({
                    "query":      query,
                    "expected":   expected,
                    "predicted":  predicted,
                    "confidence": round(confidence, 3)
                })

        accuracy = correct / len(test_cases)

        failure_msg = ""
        if failures:
            lines = "\n".join([
                f"  Query: '{f['query']}' | "
                f"Expected: {f['expected']} | "
                f"Got: {f['predicted']} ({f['confidence']})"
                for f in failures
            ])
            failure_msg = f"\n\nMisclassified:\n{lines}"

        assert accuracy >= minimum_accuracy, (
            f"Accuracy {accuracy:.1%} is below "
            f"minimum {minimum_accuracy:.1%} "
            f"({correct}/{len(test_cases)} correct)"
            f"{failure_msg}"
        )

    def test_per_intent_accuracy(self, trained_classifier, golden_data):
        """
        Each intent category should have at least 50% accuracy.
        Catches cases where overall accuracy is fine but one
        specific intent is broken.
        """
        test_cases = golden_data["test_cases"]
        by_intent  = {}

        for case in test_cases:
            intent = case["expected_intent"]
            if intent not in by_intent:
                by_intent[intent] = {"total": 0, "correct": 0}
            by_intent[intent]["total"] += 1

            predicted, _ = trained_classifier.predict(case["query"])
            if predicted == intent:
                by_intent[intent]["correct"] += 1

        for intent, counts in by_intent.items():
            accuracy = counts["correct"] / counts["total"]
            assert accuracy >= 0.5, (
                f"Intent '{intent}' accuracy is only {accuracy:.1%} "
                f"({counts['correct']}/{counts['total']})"
            )

    def test_no_none_predictions(self, trained_classifier, golden_data):
        """No query in the dataset should ever return None."""
        for case in golden_data["test_cases"]:
            intent, confidence = trained_classifier.predict(case["query"])
            assert intent     is not None
            assert confidence is not None
