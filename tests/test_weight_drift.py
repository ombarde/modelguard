"""Test Module 1: Weight Drift Analyzer."""

import torch
import torch.nn as nn
from modelguard.weight_drift import WeightDriftAnalyzer


def create_simple_model():
    """Create a simple test model."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    return model


def test_identical_models():
    """Two identical models should show zero drift."""
    model_a = create_simple_model()

    # Clone weights exactly
    model_b = create_simple_model()
    model_b.load_state_dict(model_a.state_dict())

    analyzer = WeightDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze()

    print(report.summary())

    assert report.overall_drift_score < 0.01
    assert report.architecture_match is True
    print("✅ Test 1 PASSED: Identical models show zero drift\n")


def test_modified_model():
    """Modified model should show drift."""
    model_a = create_simple_model()
    model_b = create_simple_model()
    model_b.load_state_dict(model_a.state_dict())

    # Manually modify one layer
    with torch.no_grad():
        model_b[0].weight.add_(torch.randn_like(model_b[0].weight) * 0.5)

    analyzer = WeightDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze()

    print(report.summary())

    assert report.overall_drift_score > 0.01
    print("✅ Test 2 PASSED: Modified model shows drift\n")


def test_heavily_modified_model():
    """Completely different weights should show high drift."""
    model_a = create_simple_model()
    model_b = create_simple_model()  # random new weights

    analyzer = WeightDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze()

    print(report.summary())

    assert report.overall_drift_score > 0.1
    assert len(report.most_drifted_layers) > 0
    print("✅ Test 3 PASSED: Different models show high drift\n")


def test_layer_ranking():
    """Layer ranking should work correctly."""
    model_a = create_simple_model()
    model_b = create_simple_model()
    model_b.load_state_dict(model_a.state_dict())

    # Modify only the last layer heavily
    with torch.no_grad():
        model_b[4].weight.add_(torch.randn_like(model_b[4].weight) * 2.0)

    analyzer = WeightDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze()

    ranking = report.get_layer_ranking()
    print("Layer ranking (highest drift first):")
    for name, score in ranking:
        print(f"  {name}: {score:.4f}")

    # The last layer should be ranked highest
    assert "4.weight" in ranking[0][0] or "4.bias" in ranking[0][0]
    print("\n✅ Test 4 PASSED: Layer ranking is correct\n")


if __name__ == "__main__":
    test_identical_models()
    test_modified_model()
    test_heavily_modified_model()
    test_layer_ranking()
    print("🎉 ALL MODULE 1 TESTS PASSED!")