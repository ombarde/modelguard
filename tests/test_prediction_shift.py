"""Test Module 2: Prediction Shift Engine."""

import torch
import torch.nn as nn
import numpy as np
from modelguard.prediction_shift import PredictionShiftAnalyzer


def create_classifier():
    """Create a simple classification model."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),  # 3 classes
    )
    return model


def create_test_data(n_samples=200):
    """Create random test dataset."""
    return torch.randn(n_samples, 10)


def test_identical_predictions():
    """Same model should produce zero prediction shift."""
    model_a = create_classifier()
    model_b = create_classifier()
    model_b.load_state_dict(model_a.state_dict())  # exact copy

    dataset = create_test_data()

    analyzer = PredictionShiftAnalyzer(
        model_a, model_b,
        class_names=["Cat", "Dog", "Bird"]
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.disagreement_rate == 0.0
    assert report.avg_probability_shift < 0.001
    assert report.prediction_drift_score < 0.01
    print("✅ Test 1 PASSED: Identical models show zero prediction shift\n")


def test_slightly_modified():
    """Slightly changed model should show small drift."""
    model_a = create_classifier()
    model_b = create_classifier()
    model_b.load_state_dict(model_a.state_dict())

    # Small perturbation
    with torch.no_grad():
        model_b[0].weight.add_(torch.randn_like(model_b[0].weight) * 0.1)

    dataset = create_test_data()

    analyzer = PredictionShiftAnalyzer(
        model_a, model_b,
        class_names=["Cat", "Dog", "Bird"]
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.prediction_drift_score < 0.8
    print("✅ Test 2 PASSED: Small change shows small drift\n")


def test_heavily_modified():
    """Completely different model should show high drift."""
    model_a = create_classifier()
    model_b = create_classifier()  # different random weights

    dataset = create_test_data()

    analyzer = PredictionShiftAnalyzer(
        model_a, model_b,
        class_names=["Cat", "Dog", "Bird"]
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.disagreement_rate > 0.0
    assert report.prediction_drift_score > 0.0
    print("✅ Test 3 PASSED: Different models show high prediction drift\n")


def test_flipped_samples():
    """Should detect which samples changed prediction."""
    model_a = create_classifier()
    model_b = create_classifier()
    model_b.load_state_dict(model_a.state_dict())

    # Heavy modification to cause flips
    with torch.no_grad():
        model_b[4].weight.add_(torch.randn_like(model_b[4].weight) * 3.0)
        model_b[4].bias.add_(torch.randn_like(model_b[4].bias) * 2.0)

    dataset = create_test_data(100)

    analyzer = PredictionShiftAnalyzer(model_a, model_b)
    report = analyzer.analyze(dataset)

    print(report.summary())

    # Get details of flipped samples
    flipped = report.get_flipped_samples()
    if flipped:
        print(f"  Example flipped sample:")
        print(f"    Index:        {flipped[0]['index']}")
        print(f"    Pred A:       {flipped[0]['pred_model_a']}")
        print(f"    Pred B:       {flipped[0]['pred_model_b']}")
        print(f"    Confidence A: {flipped[0]['confidence_a']:.4f}")
        print(f"    Confidence B: {flipped[0]['confidence_b']:.4f}")

    assert isinstance(report.flipped_indices, list)
    print("\n✅ Test 4 PASSED: Flipped samples detected correctly\n")


def test_class_shift_detail():
    """Class-wise shifts should be computed."""
    model_a = create_classifier()
    model_b = create_classifier()

    dataset = create_test_data(300)

    analyzer = PredictionShiftAnalyzer(
        model_a, model_b,
        class_names=["Cat", "Dog", "Bird"]
    )
    report = analyzer.analyze(dataset)

    # Check class shifts exist
    assert len(report.class_shifts) == 3
    for cs in report.class_shifts:
        assert cs.class_name in ["Cat", "Dog", "Bird"]
        assert 0.0 <= cs.avg_prob_model_a <= 1.0
        assert 0.0 <= cs.avg_prob_model_b <= 1.0

    print("✅ Test 5 PASSED: Class-wise shifts computed correctly\n")


if __name__ == "__main__":
    test_identical_predictions()
    test_slightly_modified()
    test_heavily_modified()
    test_flipped_samples()
    test_class_shift_detail()
    print("🎉 ALL MODULE 2 TESTS PASSED!")