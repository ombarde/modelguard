"""Test Module 4: Feature Drift Analyzer."""

import torch
import torch.nn as nn
import numpy as np
from modelguard.feature_drift import FeatureDriftAnalyzer, GradientAttributor


def create_model():
    """Create a simple model."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    return model


def create_test_data(n_samples=100):
    """Create random test dataset."""
    return torch.randn(n_samples, 10)


def test_gradient_attributor():
    """Test that gradient attribution works."""
    model = create_model()
    dataset = create_test_data(50)

    attributor = GradientAttributor(model)
    importance = attributor.compute_importance(dataset)

    print("Feature importance scores:")
    for i, imp in enumerate(importance):
        print(f"  Feature {i}: {imp:.4f}")

    assert len(importance) == 10
    assert abs(importance.sum() - 1.0) < 0.01  # should sum to ~1
    assert all(imp >= 0 for imp in importance)  # all non-negative
    print("\n✅ Test 1 PASSED: Gradient attribution works\n")


def test_identical_features():
    """Same model should show zero feature drift."""
    model_a = create_model()
    model_b = create_model()
    model_b.load_state_dict(model_a.state_dict())

    dataset = create_test_data()

    feature_names = [f"F_{i}" for i in range(10)]

    analyzer = FeatureDriftAnalyzer(
        model_a, model_b,
        feature_names=feature_names,
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.overall_feature_drift < 0.01
    print("✅ Test 2 PASSED: Identical models show zero feature drift\n")


def test_modified_features():
    """Modified model should show feature drift."""
    model_a = create_model()
    model_b = create_model()
    model_b.load_state_dict(model_a.state_dict())

    # Modify first layer — changes which input features matter
    with torch.no_grad():
        model_b[0].weight.add_(torch.randn_like(model_b[0].weight) * 0.5)

    dataset = create_test_data()

    feature_names = [
        "age", "income", "score", "tenure", "balance",
        "products", "active", "salary", "credit", "region"
    ]

    analyzer = FeatureDriftAnalyzer(
        model_a, model_b,
        feature_names=feature_names,
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.overall_feature_drift > 0.0
    print("✅ Test 3 PASSED: Modified model shows feature drift\n")


def test_completely_different():
    """Completely different models should show high feature drift."""
    model_a = create_model()
    model_b = create_model()  # different random weights

    dataset = create_test_data(200)

    feature_names = [
        "age", "income", "score", "tenure", "balance",
        "products", "active", "salary", "credit", "region"
    ]

    analyzer = FeatureDriftAnalyzer(
        model_a, model_b,
        feature_names=feature_names,
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    # Check ranking comparison
    comparison = report.get_feature_ranking_comparison()
    print("Feature ranking comparison:")
    for name, data in list(comparison.items())[:5]:
        print(
            f"  {name}: rank {data['rank_a']} → {data['rank_b']} "
            f"(change: {data['rank_change']:+d})"
        )

    assert report.overall_feature_drift > 0.0
    print("\n✅ Test 4 PASSED: Different models show high feature drift\n")


def test_concentration():
    """Concentration metric should work."""
    model_a = create_model()
    model_b = create_model()

    dataset = create_test_data()

    analyzer = FeatureDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze(dataset)

    assert 0.0 <= report.concentration_model_a <= 1.0
    assert 0.0 <= report.concentration_model_b <= 1.0

    print(f"Concentration Model A: {report.concentration_model_a:.4f}")
    print(f"Concentration Model B: {report.concentration_model_b:.4f}")
    print(f"Change: {report.concentration_change:+.4f}")

    print("\n✅ Test 5 PASSED: Concentration metric works\n")


if __name__ == "__main__":
    test_gradient_attributor()
    test_identical_features()
    test_modified_features()
    test_completely_different()
    test_concentration()
    print("🎉 ALL MODULE 4 TESTS PASSED!")